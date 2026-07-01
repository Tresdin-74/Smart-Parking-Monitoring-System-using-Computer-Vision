
"""
tracker.py
====================================================
این فایل مسئول تشخیص ورود/خروج خودروها از قاب دوربین و تعیین جهت حرکت آن‌هاست.

روش کار:
 1) از ماسک پیش‌زمینه (همان ماسکی که در detector.py محاسبه می‌شود) کانتورهای
    بزرگ را به‌عنوان «اشیاء متحرک» (خودرو) در نظر می‌گیریم.
 2) هر شی متحرک را با یک شناسه (ID) دنبال می‌کنیم؛ با مقایسه مرکز اشیاء در
    فریم فعلی با فریم قبلی (نزدیک‌ترین فاصله).
 3) با مقایسه موقعیت y (یا x) مرکز شی در طول زمان، جهت حرکت (بالا/پایین یا
    چپ/راست) را تشخیص می‌دهیم.
 4) از Optical Flow (روش Lucas-Kanade) برای تخمین دقیق‌تر بردار حرکت هر شی
    استفاده می‌کنیم تا جهت حرکت مطمئن‌تر باشد.
 5) وقتی مرکز یک شی از یک «خط مجازی ورودی/خروجی» عبور می‌کند، رویداد
    ورود یا خروج ثبت می‌شود.
"""

# ---------------------------------------------------------------
# وارد کردن کتابخانه‌های مورد نیاز
# ---------------------------------------------------------------
import cv2          # کتابخانه اصلی پردازش تصویر
import numpy as np  # برای محاسبات عددی


class TrackedVehicle:
    """
    این کلاس اطلاعات یک خودرو ردیابی‌شده را نگه می‌دارد:
    - شناسه یکتا
    - مرکز فعلی و قبلی
    - تعداد فریم‌هایی که از آخرین بار دیده شدن آن گذشته (برای حذف ID های قدیمی)
    - وضعیت اینکه آیا رویداد ورود/خروج آن قبلاً ثبت شده یا نه
    """

    def __init__(self, vehicle_id, center):
        self.vehicle_id = vehicle_id      # شناسه یکتای خودرو
        self.center = center               # مرکز فعلی (x, y)
        self.prev_center = center          # مرکز در فریم قبلی
        self.age = 0                       # تعداد فریم‌هایی که این شی دیده شده
        self.frames_since_seen = 0         # تعداد فریم‌های متوالی که دیده نشده
        self.crossed_line = False          # آیا از خط ورودی/خروجی عبور کرده یا نه


class VehicleTracker:
    """
    این کلاس مدیریت ردیابی چند خودرو و تشخیص ورود/خروج را انجام می‌دهد.
    """

    def __init__(self, frame_shape, line_y_ratio=0.5, max_distance=80, max_missed_frames=10):
        """
        frame_shape: ابعاد فریم (height, width, channels) برای تعیین موقعیت خط مجازی
        line_y_ratio: موقعیت خط مجازی ورود/خروج به‌صورت نسبتی از ارتفاع تصویر (0 تا 1)
                       مثلاً 0.5 یعنی وسط تصویر
        max_distance: حداکثر فاصله (پیکسل) برای تطبیق یک شی بین دو فریم متوالی
        max_missed_frames: حداکثر تعداد فریم‌هایی که یک شی می‌تواند دیده نشود
                            قبل از اینکه از لیست ردیابی حذف شود
        """
        # ارتفاع و عرض فریم
        self.frame_h, self.frame_w = frame_shape[:2]

        # موقعیت y خط مجازی ورود/خروج (یک خط افقی وسط تصویر به‌صورت پیش‌فرض)
        self.line_y = int(self.frame_h * line_y_ratio)

        # حداکثر فاصله مجاز برای تطبیق اشیاء بین فریم‌ها
        self.max_distance = max_distance

        # حداکثر فریم‌های ازدست‌رفته قبل از حذف یک شی
        self.max_missed_frames = max_missed_frames

        # دیکشنری اشیاء ردیابی‌شده: کلید = شناسه، مقدار = TrackedVehicle
        self.tracked_vehicles = {}

        # شمارنده برای تولید شناسه‌های یکتای جدید
        self.next_id = 1

        # متغیرهای لازم برای Optical Flow (فریم خاکستری قبلی)
        self.prev_gray = None

    def find_centers_from_mask(self, fg_mask, min_area=800):
        """
        این تابع از ماسک پیش‌زمینه، کانتورهای بزرگ (احتمالاً خودرو) را پیدا کرده
        و مرکز هر کدام را برمی‌گرداند.
        """
        # پیدا کردن تمام کانتورهای خارجی در ماسک
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        centers = []  # لیست مرکزهای پیدا شده

        # بررسی هر کانتور
        for contour in contours:
            # محاسبه مساحت کانتور
            area = cv2.contourArea(contour)

            # نادیده گرفتن کانتورهای کوچک (احتمالاً نویز، نه خودرو)
            if area < min_area:
                continue

            # محاسبه مستطیل محاطی برای پیدا کردن مرکز
            x, y, w, h = cv2.boundingRect(contour)
            center = (x + w // 2, y + h // 2)

            # اضافه کردن مرکز به لیست
            centers.append(center)

        return centers

    def match_centers_to_vehicles(self, centers):
        """
        این تابع مرکزهای جدید پیدا شده در فریم فعلی را با خودروهای ردیابی‌شده
        قبلی تطبیق می‌دهد (بر اساس نزدیک‌ترین فاصله). اگر مرکزی به هیچ خودرویی
        نزدیک نبود، یک خودروی جدید با شناسه جدید ساخته می‌شود.
        """
        # لیستی برای نشان دادن اینکه کدام مرکزهای جدید استفاده شده‌اند
        used_centers = set()

        # بررسی هر خودروی موجود برای پیدا کردن نزدیک‌ترین مرکز جدید
        for vehicle_id, vehicle in self.tracked_vehicles.items():
            best_match_idx = None
            best_distance = self.max_distance

            # جست‌وجو در مرکزهای جدید برای نزدیک‌ترین به مرکز قبلی این خودرو
            for idx, center in enumerate(centers):
                if idx in used_centers:
                    continue  # این مرکز قبلاً به یک خودرو دیگر اختصاص یافته

                # محاسبه فاصله اقلیدسی بین مرکز قبلی خودرو و مرکز جدید
                dist = np.sqrt((vehicle.center[0] - center[0]) ** 2 +
                                (vehicle.center[1] - center[1]) ** 2)

                # اگر این فاصله از بهترین فاصله فعلی کمتر بود، به‌روزرسانی کن
                if dist < best_distance:
                    best_distance = dist
                    best_match_idx = idx

            # اگر یک تطبیق مناسب پیدا شد
            if best_match_idx is not None:
                # ذخیره مرکز قبلی قبل از به‌روزرسانی
                vehicle.prev_center = vehicle.center

                # به‌روزرسانی مرکز فعلی با مرکز جدید
                vehicle.center = centers[best_match_idx]

                # ریست شمارنده فریم‌های ازدست‌رفته (چون این فریم دیده شد)
                vehicle.frames_since_seen = 0

                # افزایش سن (تعداد فریم‌هایی که دیده شده)
                vehicle.age += 1

                # علامت‌گذاری این مرکز به‌عنوان استفاده‌شده
                used_centers.add(best_match_idx)
            else:
                # اگر تطبیقی پیدا نشد، این خودرو در این فریم دیده نشده
                vehicle.frames_since_seen += 1

        # برای مرکزهای جدیدی که به هیچ خودرویی تطبیق نخوردند، خودروی جدید بساز
        for idx, center in enumerate(centers):
            if idx not in used_centers:
                # ساخت یک خودروی جدید با شناسه جدید
                new_vehicle = TrackedVehicle(self.next_id, center)
                self.tracked_vehicles[self.next_id] = new_vehicle

                # افزایش شمارنده شناسه برای خودروی بعدی
                self.next_id += 1

        # حذف خودروهایی که مدت زیادی دیده نشده‌اند (احتمالاً از تصویر خارج شده‌اند)
        ids_to_remove = [
            vid for vid, v in self.tracked_vehicles.items()
            if v.frames_since_seen > self.max_missed_frames
        ]
        for vid in ids_to_remove:
            del self.tracked_vehicles[vid]

    def check_line_crossing(self, vehicle):
        """
        این تابع بررسی می‌کند که آیا مرکز یک خودرو از خط مجازی عبور کرده است،
        و در این صورت تشخیص می‌دهد که این عبور به معنای «ورود» یا «خروج» است.
        برمی‌گرداند: None اگر عبوری رخ نداده، یا رشته "ENTRY"/"EXIT" در غیر این صورت.
        """
        # اگر این خودرو قبلاً عبور کرده، دوباره بررسی نکن
        if vehicle.crossed_line:
            return None

        prev_y = vehicle.prev_center[1]   # موقعیت y در فریم قبلی
        curr_y = vehicle.center[1]        # موقعیت y در فریم فعلی

        # اگر خودرو از بالا به پایین خط را رد کرد -> در نظر می‌گیریم "ورود" (ENTRY)
        if prev_y < self.line_y <= curr_y:
            vehicle.crossed_line = True
            return "ENTRY"

        # اگر خودرو از پایین به بالا خط را رد کرد -> در نظر می‌گیریم "خروج" (EXIT)
        if prev_y > self.line_y >= curr_y:
            vehicle.crossed_line = True
            return "EXIT"

        # اگر هیچ عبوری رخ نداده
        return None

    def estimate_direction_with_optical_flow(self, gray_frame):
        """
        این تابع با استفاده از Optical Flow متراکم (Dense Optical Flow - Farneback)
        یک تخمین کلی از جهت حرکت در صحنه به دست می‌آورد.
        خروجی: یک تصویر رنگی که جهت/سرعت حرکت را به‌صورت رنگ نشان می‌دهد (برای دیباگ/نمایش)
        و بردار میانگین حرکت (dx, dy).
        """
        # اگر فریم خاکستری قبلی موجود نباشد (اولین فریم)، optical flow قابل محاسبه نیست
        if self.prev_gray is None:
            self.prev_gray = gray_frame
            return None, (0, 0)

        # محاسبه Optical Flow متراکم با الگوریتم Farneback
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray_frame, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )

        # میانگین بردار حرکت در کل تصویر (dx, dy)
        mean_dx = float(np.mean(flow[..., 0]))
        mean_dy = float(np.mean(flow[..., 1]))

        # به‌روزرسانی فریم خاکستری قبلی برای فریم بعد
        self.prev_gray = gray_frame

        return flow, (mean_dx, mean_dy)

    def update(self, fg_mask, gray_frame):
        """
        تابع اصلی به‌روزرسانی ردیاب: باید برای هر فریم صدا زده شود.
        ورودی‌ها:
            fg_mask: ماسک پیش‌زمینه (از detector.py)
            gray_frame: فریم سطح خاکستری فعلی (برای optical flow)
        خروجی:
            لیستی از رویدادهای ورود/خروج که در این فریم رخ داده‌اند
            و دیکشنری خودروهای فعلی ردیابی‌شده (برای رسم روی تصویر)
        """
        # پیدا کردن مرکزهای اشیاء متحرک از ماسک پیش‌زمینه
        centers = self.find_centers_from_mask(fg_mask)

        # تطبیق مرکزهای جدید با خودروهای ردیابی‌شده موجود (یا ساخت خودرو جدید)
        self.match_centers_to_vehicles(centers)

        # تخمین جهت کلی حرکت صحنه با Optical Flow (اطلاعات کمکی)
        _, mean_flow = self.estimate_direction_with_optical_flow(gray_frame)

        # لیستی برای رویدادهای ورود/خروج این فریم
        events = []

        # بررسی عبور هر خودرو از خط مجازی
        for vehicle_id, vehicle in self.tracked_vehicles.items():
            crossing = self.check_line_crossing(vehicle)

            # اگر عبوری رخ داده بود، رویداد را به لیست اضافه کن
            if crossing is not None:
                events.append({
                    "vehicle_id": vehicle_id,
                    "event": crossing,         # "ENTRY" یا "EXIT"
                    "position": vehicle.center,
                    "mean_flow": mean_flow      # بردار میانگین حرکت برای اطلاعات بیشتر
                })

        return events, self.tracked_vehicles
