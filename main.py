
"""
main.py
====================================================
فایل اصلی اجرای سیستم نظارت تصویری هوشمند پارکینگ.

این فایل تمام بخش‌های دیگر پروژه را به هم متصل می‌کند:
 - detector.py   : تشخیص اشغال/خالی بودن هر جای پارک
 - tracker.py    : تشخیص ورود/خروج خودرو و جهت حرکت
 - logger.py     : ثبت رویدادها در فایل‌های CSV با timestamp
 - utils.py      : توابع کمکی (CLAHE و ...)

همچنین یک «داشبورد بصری آنی» در گوشه تصویر رسم می‌کند که وضعیت کلی
پارکینگ (تعداد جای خالی/اشغال) را نمایش می‌دهد، و خروجی نهایی (ویدئو
با تمام overlay ها) را ذخیره می‌کند.

نحوه اجرا:
    1) ابتدا با اجرای spot_selector.py، جاهای پارک را روی یک فریم نمونه
       مشخص و در فایل parking_spots.json ذخیره کنید.
    2) سپس این فایل (main.py) را اجرا کنید.

برای جزئیات کامل به README.md مراجعه کنید.
"""

# ---------------------------------------------------------------
# وارد کردن کتابخانه‌های مورد نیاز
# ---------------------------------------------------------------
import cv2          # کتابخانه اصلی پردازش تصویر
import json         # برای خواندن فایل مختصات جاهای پارک
import numpy as np  # برای محاسبات عددی

# وارد کردن ماژول‌های داخلی پروژه
from detector import ParkingDetector
from tracker import VehicleTracker
from logger import EventLogger

# ---------------------------------------------------------------
# تنظیمات اولیه
# ---------------------------------------------------------------
VIDEO_SOURCE = "parking_video.mp4"        # مسیر ویدئوی ورودی (یا 0 برای وبکم)
SPOTS_JSON_PATH = "parking_spots.json"    # فایل مختصات جاهای پارک (خروجی spot_selector.py)
OUTPUT_VIDEO_PATH = "parking_output.avi"  # مسیر ذخیره ویدئوی خروجی پردازش‌شده


def load_parking_spots(json_path):
    """
    این تابع لیست جاهای پارک را از فایل JSON می‌خواند.
    هر جای پارک یک لیست از 4 نقطه (x, y) است.
    """
    # باز کردن و خواندن فایل JSON
    with open(json_path, "r", encoding="utf-8") as f:
        spots = json.load(f)

    return spots


def draw_parking_spots(frame, spot_results, detector):
    """
    این تابع روی فریم، چندضلعی هر جای پارک را با رنگ مناسب رسم می‌کند:
      - قرمز: اشغال
      - سبز: خالی
    و شماره هر جای پارک را داخل آن می‌نویسد.
    """
    # حلقه روی نتایج هر جای پارک
    for result in spot_results:
        # پیدا کردن شیء ParkingSpot متناظر برای دسترسی به نقاط چندضلعی
        spot_obj = next(s for s in detector.spots if s.spot_id == result["spot_id"])

        # انتخاب رنگ بر اساس وضعیت: قرمز برای اشغال، سبز برای خالی
        color = (0, 0, 255) if result["occupied"] else (0, 255, 0)

        # رسم چندضلعی دور جای پارک با رنگ مشخص‌شده
        cv2.polylines(frame, [spot_obj.points], isClosed=True, color=color, thickness=2)

        # محاسبه مرکز تقریبی جای پارک برای قرار دادن متن شماره
        cx = int(np.mean(spot_obj.points[:, 0]))
        cy = int(np.mean(spot_obj.points[:, 1]))

        # نوشتن شماره جای پارک
        cv2.putText(frame, str(result["spot_id"]), (cx - 10, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def draw_dashboard(frame, spot_results, vehicle_events_count):
    """
    این تابع یک پنل داشبورد نیمه‌شفاف در گوشه بالا-سمت‌چپ تصویر رسم می‌کند
    که شامل تعداد کل جاها، تعداد خالی، تعداد اشغال، و تعداد کل رویدادهای
    ورود/خروج ثبت‌شده تا این لحظه است.
    """
    # محاسبه آمار کلی
    total_spots = len(spot_results)
    occupied_spots = sum(1 for r in spot_results if r["occupied"])
    empty_spots = total_spots - occupied_spots

    # ابعاد پنل داشبورد
    panel_w, panel_h = 260, 110

    # رسم پس‌زمینه نیمه‌شفاف پنل
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), (40, 40, 40), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # نوشتن عنوان داشبورد
    cv2.putText(frame, "Parking Dashboard", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    # نوشتن آمار جاها
    cv2.putText(frame, f"Total Spots: {total_spots}", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(frame, f"Empty: {empty_spots}", (20, 82),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    cv2.putText(frame, f"Occupied: {occupied_spots}", (140, 82),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    # نوشتن تعداد رویدادهای ورود/خروج ثبت‌شده
    cv2.putText(frame, f"Vehicle Events: {vehicle_events_count}", (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)


def draw_tracked_vehicles(frame, tracked_vehicles, line_y):
    """
    این تابع موقعیت خودروهای ردیابی‌شده را با یک دایره و شناسه‌شان نشان می‌دهد
    و خط مجازی ورود/خروج را روی تصویر رسم می‌کند.
    """
    # رسم خط مجازی افقی ورود/خروج (به رنگ زرد)
    cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 255, 255), 2)
    cv2.putText(frame, "Entry/Exit Line", (10, line_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # رسم هر خودروی ردیابی‌شده
    for vehicle_id, vehicle in tracked_vehicles.items():
        # رسم دایره روی مرکز خودرو
        cv2.circle(frame, vehicle.center, 6, (255, 0, 255), -1)

        # نوشتن شناسه خودرو کنار دایره
        cv2.putText(frame, f"ID {vehicle_id}", (vehicle.center[0] + 8, vehicle.center[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)


def main():
    """
    تابع اصلی برنامه: حلقه پردازش ویدئو و فراخوانی تمام ماژول‌ها.
    """
    # خواندن مختصات جاهای پارک از فایل JSON
    try:
        spots_points_list = load_parking_spots(SPOTS_JSON_PATH)
    except FileNotFoundError:
        print(f"خطا: فایل '{SPOTS_JSON_PATH}' پیدا نشد.")
        print("لطفاً ابتدا spot_selector.py را اجرا کرده و جاهای پارک را تعیین کنید.")
        return

    # باز کردن منبع ویدئو
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print("خطا: امکان باز کردن ویدئو وجود ندارد.")
        return

    # خواندن یک فریم نمونه برای گرفتن ابعاد تصویر
    ret, sample_frame = cap.read()
    if not ret:
        print("خطا: امکان خواندن فریم اولیه وجود ندارد.")
        return

    frame_h, frame_w = sample_frame.shape[:2]

    # بازگرداندن موقعیت ویدئو به فریم اول (چون یک فریم برای نمونه خوانده شد)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # ساخت تشخیص‌دهنده وضعیت جاهای پارک
    detector = ParkingDetector(spots_points_list, motion_threshold_ratio=0.15, vote_threshold=8)

    # ساخت ردیاب خودروها (خط مجازی در وسط ارتفاع تصویر)
    tracker = VehicleTracker(sample_frame.shape, line_y_ratio=0.5)

    # ساخت سیستم ثبت رویدادها در CSV
    logger = EventLogger()

    # تعریف VideoWriter برای ذخیره ویدئوی خروجی
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out_writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, 20.0, (frame_w, frame_h))

    # شمارنده کل رویدادهای ورود/خروج برای نمایش در داشبورد
    vehicle_events_count = 0

    # حلقه اصلی پردازش ویدئو
    while True:
        # خواندن یک فریم جدید
        ret, frame = cap.read()
        if not ret:
            # اگر ویدئو تمام شد، از حلقه خارج شو
            break

        # به‌روزرسانی تشخیص وضعیت جاهای پارک (بخش 1: occupancy detection)
        spot_results, fg_mask = detector.update(frame)

        # تبدیل فریم فعلی به سطح خاکستری برای optical flow در tracker
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # به‌روزرسانی ردیاب خودروها (بخش 2: ورود/خروج و جهت حرکت)
        vehicle_events, tracked_vehicles = tracker.update(fg_mask, gray_frame)

        # ------ ثبت رویدادهای تغییر وضعیت جاهای پارک در CSV ------
        for result in spot_results:
            if result["status_changed"]:
                status_str = "OCCUPIED" if result["occupied"] else "EMPTY"
                logger.log_spot_event(result["spot_id"], status_str)

        # ------ ثبت رویدادهای ورود/خروج خودرو در CSV ------
        for event in vehicle_events:
            logger.log_vehicle_event(event["vehicle_id"], event["event"], event["position"])
            vehicle_events_count += 1

        # ------ رسم تمام overlay ها روی فریم ------
        # رسم چندضلعی جاهای پارک با رنگ متناسب با وضعیت
        draw_parking_spots(frame, spot_results, detector)

        # رسم خودروهای ردیابی‌شده و خط مجازی ورود/خروج
        draw_tracked_vehicles(frame, tracked_vehicles, tracker.line_y)

        # رسم داشبورد آماری در گوشه تصویر
        draw_dashboard(frame, spot_results, vehicle_events_count)

        # ------ نمایش پنجره‌ها ------
        cv2.imshow("Parking Surveillance System", frame)  # تصویر نهایی با overlay
        cv2.imshow("Foreground Mask", fg_mask)              # ماسک حرکت برای دیباگ

        # ------ ذخیره فریم در ویدئوی خروجی ------
        out_writer.write(frame)

        # خروج با کلید 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # ---------------- آزادسازی منابع ----------------
    cap.release()
    out_writer.release()
    cv2.destroyAllWindows()

    print(f"ویدئوی خروجی در '{OUTPUT_VIDEO_PATH}' ذخیره شد.")
    print("فایل‌های لاگ 'vehicle_events.csv' و 'spot_events.csv' به‌روزرسانی شدند.")


# ---------------------------------------------------------------
# نقطه شروع اجرای برنامه
# ---------------------------------------------------------------
if __name__ == "__main__":
    main()
