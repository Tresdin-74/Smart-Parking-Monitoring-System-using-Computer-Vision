
"""
logger.py
====================================================
این فایل مسئول ثبت تمام رویدادهای سیستم در فایل‌های CSV است:
 - رویدادهای ورود/خروج خودرو (با timestamp و موقعیت)
 - رویدادهای تغییر وضعیت جاهای پارک (پارک شدن / ترک کردن، با timestamp)

هر رویداد به‌محض وقوع، با timestamp دقیق در فایل مربوطه نوشته می‌شود.
"""

# ---------------------------------------------------------------
# وارد کردن کتابخانه‌های مورد نیاز
# ---------------------------------------------------------------
import csv                      # برای نوشتن فایل‌های CSV
import os                        # برای بررسی وجود فایل (جهت نوشتن هدر فقط یک‌بار)
from utils import get_timestamp  # تابع کمکی برای گرفتن زمان فعلی


class EventLogger:
    """
    این کلاس دو فایل CSV را مدیریت می‌کند:
      1) vehicle_events.csv  -> رویدادهای ورود/خروج خودرو
      2) spot_events.csv     -> رویدادهای پارک/ترک هر جای پارک
    """

    def __init__(self, vehicle_log_path="vehicle_events.csv", spot_log_path="spot_events.csv"):
        # مسیر فایل لاگ رویدادهای خودرو
        self.vehicle_log_path = vehicle_log_path

        # مسیر فایل لاگ رویدادهای جای پارک
        self.spot_log_path = spot_log_path

        # ساخت فایل‌ها با هدر مناسب در صورت عدم وجود
        self._init_csv(self.vehicle_log_path, ["Timestamp", "VehicleID", "Event", "X", "Y"])
        self._init_csv(self.spot_log_path, ["Timestamp", "SpotID", "Status"])

    def _init_csv(self, path, header):
        """
        این تابع داخلی، اگر فایل CSV از قبل وجود نداشته باشد، آن را با
        سطر هدر مشخص‌شده می‌سازد. اگر فایل وجود داشته باشد، دست‌نخورده می‌ماند
        (تا داده‌های قبلی پاک نشوند).
        """
        # بررسی اینکه آیا فایل از قبل وجود دارد
        if not os.path.exists(path):
            # ساخت فایل جدید و نوشتن سطر هدر
            with open(path, mode="w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow(header)

    def log_vehicle_event(self, vehicle_id, event_type, position):
        """
        ثبت یک رویداد ورود/خروج خودرو.
        event_type: "ENTRY" یا "EXIT"
        position: تاپل (x, y) موقعیت خودرو در لحظه عبور
        """
        # باز کردن فایل در حالت append (اضافه کردن به انتهای فایل)
        with open(self.vehicle_log_path, mode="a", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)

            # نوشتن یک سطر جدید با timestamp فعلی
            writer.writerow([get_timestamp(), vehicle_id, event_type, position[0], position[1]])

    def log_spot_event(self, spot_id, status):
        """
        ثبت یک رویداد تغییر وضعیت جای پارک.
        status: "OCCUPIED" (پارک شد) یا "EMPTY" (ترک کرد)
        """
        # باز کردن فایل در حالت append
        with open(self.spot_log_path, mode="a", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)

            # نوشتن یک سطر جدید با timestamp فعلی
            writer.writerow([get_timestamp(), spot_id, status])
