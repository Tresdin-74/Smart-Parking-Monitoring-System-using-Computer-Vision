
"""
spot_selector.py
====================================================
این فایل یک ابزار کمکی است که با استفاده از Mouse Callback به کاربر اجازه می‌دهد
محل دقیق هر جای پارک را روی یک فریم نمونه از ویدئو، با کلیک چهار نقطه (چهارضلعی)
مشخص کند. مختصات نهایی در یک فایل JSON ذخیره می‌شوند تا توسط detector.py
و main.py استفاده شوند.

نحوه استفاده:
    python spot_selector.py

راهنمای کلیدها:
    - کلیک چپ: اضافه کردن یک نقطه به جای پارک فعلی
    - کلید 'n': اتمام جای پارک فعلی و شروع جای پارک بعدی
    - کلید 's': ذخیره تمام جاهای پارک در فایل JSON و خروج
    - کلید 'q': خروج بدون ذخیره
"""

# ---------------------------------------------------------------
# وارد کردن کتابخانه‌های مورد نیاز
# ---------------------------------------------------------------
import cv2     # کتابخانه اصلی پردازش تصویر
import json    # برای ذخیره مختصات جاهای پارک در فایل JSON

# ---------------------------------------------------------------
# تنظیمات اولیه
# ---------------------------------------------------------------
VIDEO_SOURCE = "parking_video.mp4"     # مسیر ویدئوی پارکینگ (یا 0 برای وبکم)
OUTPUT_JSON_PATH = "parking_spots.json"  # مسیر ذخیره مختصات جاهای پارک

# لیست تمام جاهای پارک؛ هر جای پارک خودش یک لیست از 4 نقطه (x, y) است
all_spots = []

# لیست نقاط جای پارک فعلی که در حال انتخاب است
current_spot_points = []


def mouse_callback(event, x, y, flags, param):
    """
    این تابع هر بار که کاربر روی پنجره تصویر کلیک می‌کند صدا زده می‌شود.
    با کلیک چپ، نقطه (x, y) به جای پارک فعلی اضافه می‌شود.
    """
    # بررسی اینکه آیا کلیک، کلیک چپ ماوس بوده است
    if event == cv2.EVENT_LBUTTONDOWN:
        # اضافه کردن نقطه کلیک‌شده به لیست نقاط جای پارک فعلی
        current_spot_points.append((x, y))

        # چاپ نقطه اضافه‌شده برای اطلاع کاربر
        print(f"نقطه اضافه شد: ({x}, {y}) -- تعداد نقاط فعلی: {len(current_spot_points)}")


def draw_existing_spots(frame):
    """
    این تابع تمام جاهای پارک تعریف‌شده تا این لحظه (هم جاهای کامل‌شده،
    هم نقاط جای پارک فعلی) را روی فریم رسم می‌کند تا کاربر وضعیت را ببیند.
    """
    # رسم تمام جاهای پارک کامل‌شده (هر کدام یک چندضلعی بسته با رنگ سبز)
    for idx, spot in enumerate(all_spots):
        # تبدیل لیست نقاط به فرمت مناسب برای cv2.polylines
        pts = [list(p) for p in spot]

        # رسم چندضلعی بسته با رنگ سبز
        cv2.polylines(frame, [cv2_points(pts)], isClosed=True, color=(0, 255, 0), thickness=2)

        # نوشتن شماره جای پارک در مرکز تقریبی آن
        cx = sum(p[0] for p in spot) // len(spot)
        cy = sum(p[1] for p in spot) // len(spot)
        cv2.putText(frame, str(idx + 1), (cx - 10, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # رسم نقاط جای پارک فعلی (در حال انتخاب) با رنگ زرد
    for point in current_spot_points:
        cv2.circle(frame, point, 4, (0, 255, 255), -1)

    # اگر بیش از یک نقطه برای جای پارک فعلی انتخاب شده، خطوط بین آن‌ها را رسم کن
    if len(current_spot_points) > 1:
        cv2.polylines(frame, [cv2_points(current_spot_points)], isClosed=False,
                       color=(0, 255, 255), thickness=2)


def cv2_points(points_list):
    """
    این تابع یک لیست ساده از تاپل‌های (x, y) را به فرمت آرایه numpy مناسب
    برای توابع cv2.polylines / cv2.fillPoly تبدیل می‌کند.
    """
    import numpy as np  # وارد کردن numpy فقط در صورت نیاز این تابع
    return np.array(points_list, dtype=np.int32)


def main():
    """
    تابع اصلی: یک فریم نمونه از ویدئو می‌گیرد و رابط انتخاب جاهای پارک را اجرا می‌کند.
    """
    # باز کردن منبع ویدئو (فایل یا وبکم)
    cap = cv2.VideoCapture(VIDEO_SOURCE)

    # بررسی موفقیت باز شدن ویدئو
    if not cap.isOpened():
        print("خطا: امکان باز کردن ویدئو وجود ندارد.")
        return

    # خواندن یک فریم نمونه (اولین فریم) برای انتخاب جاهای پارک روی آن
    ret, base_frame = cap.read()
    if not ret:
        print("خطا: امکان خواندن فریم از ویدئو وجود ندارد.")
        return

    # رها کردن منبع ویدئو چون فقط به یک فریم نیاز داشتیم
    cap.release()

    # ساخت پنجره و تنظیم تابع callback ماوس روی آن
    window_name = "Parking Spot Selector"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    # حلقه اصلی برنامه
    while True:
        # کپی از فریم پایه تا تغییرات روی نسخه جدید اعمال شود (فریم اصلی دست‌نخورده بماند)
        display_frame = base_frame.copy()

        # رسم راهنما روی تصویر
        cv2.putText(display_frame,
                    "Click 4 corners | 'n': next spot | 's': save | 'q': quit",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # رسم تمام جاهای پارک تا این لحظه
        draw_existing_spots(display_frame)

        # نمایش فریم
        cv2.imshow(window_name, display_frame)

        # خواندن کلید فشرده‌شده
        key = cv2.waitKey(20) & 0xFF

        # اگر کلید 'n' فشرده شد و دقیقاً 4 نقطه انتخاب شده بود، این جای پارک را ثبت کن
        if key == ord('n'):
            if len(current_spot_points) == 4:
                # اضافه کردن جای پارک فعلی به لیست کل جاهای پارک
                all_spots.append(current_spot_points.copy())
                print(f"جای پارک شماره {len(all_spots)} ثبت شد.")

                # خالی کردن لیست نقاط برای جای پارک بعدی
                current_spot_points.clear()
            else:
                print("برای ثبت یک جای پارک باید دقیقاً 4 نقطه انتخاب شود.")

        # اگر کلید 's' فشرده شد، تمام جاهای پارک را در فایل JSON ذخیره کن و خارج شو
        elif key == ord('s'):
            with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
                json.dump(all_spots, f, ensure_ascii=False, indent=2)
            print(f"تعداد {len(all_spots)} جای پارک در '{OUTPUT_JSON_PATH}' ذخیره شد.")
            break

        # اگر کلید 'q' فشرده شد، بدون ذخیره خارج شو
        elif key == ord('q'):
            print("خروج بدون ذخیره.")
            break

    # بستن تمام پنجره‌ها
    cv2.destroyAllWindows()


# ---------------------------------------------------------------
# نقطه شروع اجرای برنامه
# ---------------------------------------------------------------
if __name__ == "__main__":
    main()
