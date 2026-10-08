"""Create the configured MySQL database and batch detection tables if absent."""
import sys
from pathlib import Path

import pymysql

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from db import DB_CONFIG


def initialize_database():
    config = dict(DB_CONFIG)
    database = config.pop('database')
    identifier = '`' + database.replace('`', '``') + '`'
    connection = pymysql.connect(**config)
    try:
        with connection.cursor() as cursor:
            cursor.execute(f'CREATE DATABASE IF NOT EXISTS {identifier} CHARACTER SET utf8mb4')
            cursor.execute(f'USE {identifier}')
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS batch_tasks (
                    task_id CHAR(36) PRIMARY KEY,
                    task_name VARCHAR(255) NOT NULL,
                    status VARCHAR(20) NOT NULL DEFAULT 'running',
                    total_count INT NOT NULL DEFAULT 0,
                    completed_count INT NOT NULL DEFAULT 0,
                    ai_count INT NOT NULL DEFAULT 0,
                    suspected_count INT NOT NULL DEFAULT 0,
                    real_count INT NOT NULL DEFAULT 0,
                    error_count INT NOT NULL DEFAULT 0,
                    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    INDEX idx_tasks_created_at (created_at)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS batch_images (
                    image_id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
                    task_id CHAR(36) NOT NULL,
                    filename VARCHAR(255) NOT NULL,
                    saved_path TEXT NOT NULL,
                    status VARCHAR(20) NOT NULL DEFAULT 'pending',
                    label VARCHAR(32) NULL,
                    label_text VARCHAR(255) NULL,
                    confidence DOUBLE NULL,
                    npr_result JSON NULL,
                    deepfake_result JSON NULL,
                    tamper_result JSON NULL,
                    specialized_models JSON NULL,
                    combined_result JSON NULL,
                    deepfake_combined_result JSON NULL,
                    tamper_combined_result JSON NULL,
                    final_comparison_result JSON NULL,
                    watermark_result JSON NULL,
                    content_analysis JSON NULL,
                    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    INDEX idx_images_task (task_id, image_id),
                    CONSTRAINT fk_images_task FOREIGN KEY (task_id) REFERENCES batch_tasks (task_id)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
        connection.commit()
        print(f'MySQL database {database} is ready.')
    finally:
        connection.close()


if __name__ == '__main__':
    initialize_database()
