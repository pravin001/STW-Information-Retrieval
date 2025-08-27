import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import schedule
import time
from crawler.crawler import crawl

def scheduled_job():
    print("Scheduler is running - Starting crawler job...")
    crawl()

print("Initializing scheduler...")
schedule.every().tuesday.at("02:00").do(scheduled_job)
print(f"Next job is scheduled to run at: {schedule.next_run()}")

print("Scheduler is now running...")
while True:
    schedule.run_pending()
    time.sleep(1)
    print("Waiting for next scheduled run...", end="\r")