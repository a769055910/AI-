# -*- coding: utf-8 -*-
"""
数据库连接模块 — pymysql + ai_picture
"""
import json
import os
from pathlib import Path

import pymysql

_config_path = Path(__file__).with_name('mysql.local.json')
_local_config = json.loads(_config_path.read_text(encoding='utf-8')) if _config_path.exists() else {}

DB_CONFIG = {
    'host': os.environ.get('MYSQL_HOST', _local_config.get('host', '127.0.0.1')),
    'port': int(os.environ.get('MYSQL_PORT', _local_config.get('port', 3306))),
    'user': os.environ.get('MYSQL_USER', _local_config.get('user', 'root')),
    'password': os.environ.get('MYSQL_PASSWORD', _local_config.get('password', '')),
    'database': os.environ.get('MYSQL_DATABASE', _local_config.get('database', 'ai_picture')),
    'connect_timeout': 5,
    'charset': 'utf8mb4',
    'cursorclass': pymysql.cursors.DictCursor,
}


def get_db():
    """获取数据库连接"""
    return pymysql.connect(**DB_CONFIG)
