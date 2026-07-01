
"""
detector.py
====================================================
این فایل مسئول تشخیص وضعیت «اشغال / خالی» هر جای پارک است.

روش کار:
 1) از هر فریم، با CLAHE کنتراست را بهبود می‌دهیم تا در برابر تغییرات نور
    مقاوم‌تر باشیم.
 2) با Background Subtraction (MOG2) ماسک حرکت/تغییر را به دست می‌آوریم.
 3) برای هر جای پارک (که با یک چندضلعی 4 نقطه‌ای مشخص شده)، درصد پیکسل‌های
    «تغییر کرده» داخل آن ناحیه را محاسبه می‌کنیم.
 4) همچنین هیستوگرام ناحیه فعلی را با هیستوگرام حالت «خالی» (مرجع) مقایسه
    می‌کنیم تا تشخیص دقیق‌تر شود.
 5) برای مقابله با flickering (تغییر وضعیت سریع و موقت)، یک شمارنده رأی‌گیری
    (voting counter) برای هر جای پارک نگه می‌داریم: وضعیت فقط زمانی واقعاً
    تغییر می‌کند که چند فریم متوالی همان وضعیت جدید را تأیید کنند.
"""

# ---------------------------------------------------------------
# وارد کردن کتابخانه‌های مورد نیاز
# ---------------------------------------------------------------
import cv2                              # کتابخانه اصلی پردازش تصویر
import numpy as np                      # برای محاسبات عددی و ماسک‌ها
from utils import apply_clahe, compare_histograms  # توابع کمکی


class ParkingSpot:
    """
    این کلاس اطلاعات و وضعیت یک جای پارک را نگه می‌دارد:
    - نقاط چندضلعی تعریف‌کننده محدوده جای پارک
    - وضعیت فعلی (اشغال/خالی)
    - شمارنده رأی‌گیری برای مقابله با flickering
    - تصویر مرجع (هنگامی که جای پارک خالی است) برای مقایسه هیستوگرام
    """

    def __init__(self, spot_id, points):
        # شماره (شناسه) جای پارک
        self.spot_id = spot_id

        # نقاط چندضلعی (لیست تاپل‌های x,y) که محدوده جای پارک را مشخص می‌کنند
        self.points = np.array(points, dtype=np.int32)

        # وضعیت فعلی: False = خالی، True = اشغال
        self.occupied = False

        # شمارنده رأی برای وضعیت «اشغال» (هرچه بیشتر شود، اطمینان بیشتر است)
        self.occupied_votes = 0

        # شمارنده رأی برای وضعیت «خالی»
        self.empty_votes = 0

        # تصویر مرجع از حالت خالی (در اولین فریم‌ها ساخته می‌شود)
        self.reference_patch = None

    def get_bounding_rect(self):
        """
        این تابع کوچک‌ترین مستطیل محاطی (bounding rect) دور چندضلعی جای پارک
        را برمی‌گرداند؛ برای استخراج ناحیه (patch) از تصویر استفاده می‌شود.
        """
        # cv2.boundingRect مستطیل (x, y, w, h) را برمی‌گرداند
        return cv2.boundingRect(self.points)

    def get_mask(self, frame_shape):
        """
        این تابع یک ماسک باینری هم‌اندازه فریم می‌سازد که فقط ناحیه داخل
        چندضلعی این جای پارک سفید (255) است؛ برای محاسبه دقیق پیکسل‌های
        داخل جای پارک استفاده می‌شود.
        """
        # ساخت یک تصویر سیاه هم‌اندازه فریم
        mask = np.zeros(frame_shape[:2], dtype=np.uint8)

        # پر کردن ناحیه چندضلعی با رنگ سفید (255)
        cv2.fillPoly(mask, [self.points], 255)

        return mask


class ParkingDetector:
    """
    این کلاس اصلی تشخیص وضعیت تمام جاهای پارک را مدیریت می‌کند.
    """

    def __init__(self, spots_points_list,
                 motion_threshold_ratio=0.15,
                 vote_threshold=8):
        """
        spots_points_list: لیستی از لیست نقاط هر جای پارک (خروجی spot_selector.py)
        motion_threshold_ratio: نسبت پیکسل‌های متحرک به کل پیکسل‌های جای پارک
                                  که بالاتر از آن، جای پارک «اشغال» در نظر گرفته می‌شود
        vote_threshold: تعداد فریم‌های متوالی لازم برای تأیید تغییر وضعیت
                        (برای کاهش flickering)
        """
        # ساخت یک شیء ParkingSpot برای هر جای پارک
        self.spots = [ParkingSpot(idx + 1, pts) for idx, pts in enumerate(spots_points_list)]

        # آستانه نسبت حرکت برای تشخیص اشغال
        self.motion_threshold_ratio = motion_threshold_ratio

        # تعداد رأی لازم برای تغییر وضعیت (مقابله با flicker)
        self.vote_threshold = vote_threshold

        # ساخت آبجکت Background Subtractor با الگوریتم MOG2
        # history: تعداد فریم‌هایی که برای ساخت مدل پس‌زمینه استفاده می‌شود
        # varThreshold: آستانه واریانس برای تشخیص پیش‌زمینه/پس‌زمینه
        # detectShadows=True: تشخیص و حذف سایه‌ها (که می‌توانند باعث خطا شوند)
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=True
        )

    def preprocess_frame(self, frame):
        """
        این تابع فریم ورودی را برای پردازش آماده می‌کند:
        1) تبدیل به سطح خاکستری
        2) اعمال CLAHE برای مقاومت در برابر تغییرات نور
        """
        # تبدیل به سطح خاکستری
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # اعمال CLAHE برای بهبود کنتراست محلی
        enhanced = apply_clahe(gray)

        return enhanced

    def get_foreground_mask(self, frame):
        """
        این تابع با استفاده از Background Subtractor، ماسک پیش‌زمینه
        (پیکسل‌های متفاوت از پس‌زمینه/مدل) را محاسبه می‌کند.
        """
        # محاسبه ماسک پیش‌زمینه با MOG2
        fg_mask = self.bg_subtractor.apply(frame)

        # حذف سایه‌ها: در MOG2 با detectShadows=True، سایه‌ها مقدار 127 دارند
        # و پیکسل‌های پیش‌زمینه واقعی مقدار 255 دارند. فقط 255 را نگه می‌داریم.
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # عملیات مورفولوژیکی برای حذف نویزهای ریز
        kernel = np.ones((3, 3), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        return fg_mask

    def update(self, frame):
        """
        این تابع باید برای هر فریم جدید صدا زده شود.
        وضعیت تمام جاهای پارک را به‌روزرسانی می‌کند و لیست به‌روزشده spots
        را برمی‌گرداند. هر آیتم شامل: شناسه، وضعیت فعلی، و اینکه آیا
        وضعیت در این فریم تغییر کرده است یا نه (برای ثبت در لاگ).
        """
        # پیش‌پردازش فریم (سطح خاکستری + CLAHE)
        enhanced_gray = self.preprocess_frame(frame)

        # تبدیل تصویر بهبودیافته به BGR ساده تا با MOG2 (که روی فریم رنگی کار می‌کند) سازگار شود
        enhanced_bgr = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)

        # گرفتن ماسک پیش‌زمینه (حرکت/تغییر) از کل فریم
        fg_mask = self.get_foreground_mask(enhanced_bgr)

        # لیستی برای ذخیره نتایج به‌روزشده هر جای پارک
        results = []

        # بررسی هر جای پارک به‌صورت جدا
        for spot in self.spots:
            # ساخت ماسک مخصوص این جای پارک (فقط ناحیه داخل چندضلعی)
            spot_mask = spot.get_mask(frame.shape)

            # ترکیب ماسک پیش‌زمینه با ماسک جای پارک: فقط پیکسل‌های متحرک *داخل* این جای پارک
            overlap = cv2.bitwise_and(fg_mask, spot_mask)

            # تعداد کل پیکسل‌های داخل این جای پارک
            total_pixels = cv2.countNonZero(spot_mask)

            # تعداد پیکسل‌های متحرک داخل این جای پارک
            motion_pixels = cv2.countNonZero(overlap)

            # محاسبه نسبت پیکسل‌های متحرک به کل پیکسل‌های جای پارک
            motion_ratio = motion_pixels / float(total_pixels) if total_pixels > 0 else 0.0

            # تشخیص اولیه بر اساس نسبت حرکت: اگر بیشتر از آستانه بود، احتمالاً اشغال است
            raw_occupied_guess = motion_ratio > self.motion_threshold_ratio

            # ------ رأی‌گیری برای مقابله با flickering ------
            if raw_occupied_guess:
                # اگر این فریم «اشغال» تشخیص داده شد، شمارنده اشغال را زیاد کن
                # و شمارنده خالی را صفر کن
                spot.occupied_votes += 1
                spot.empty_votes = 0
            else:
                # اگر این فریم «خالی» تشخیص داده شد، برعکس عمل کن
                spot.empty_votes += 1
                spot.occupied_votes = 0

            # وضعیت قبلی را نگه می‌داریم تا تشخیص تغییر وضعیت ممکن شود
            previous_status = spot.occupied

            # فقط زمانی وضعیت را عوض کن که شمارنده به آستانه رأی‌گیری برسد
            if spot.occupied_votes >= self.vote_threshold:
                spot.occupied = True
            elif spot.empty_votes >= self.vote_threshold:
                spot.occupied = False
            # در غیر این صورت (هنوز رأی کافی جمع نشده)، وضعیت قبلی حفظ می‌شود

            # بررسی اینکه آیا در همین فریم وضعیت واقعاً تغییر کرده است
            status_changed = (previous_status != spot.occupied)

            # اضافه کردن نتیجه این جای پارک به لیست خروجی
            results.append({
                "spot_id": spot.spot_id,
                "occupied": spot.occupied,
                "status_changed": status_changed,
                "motion_ratio": motion_ratio
            })

        return results, fg_mask
