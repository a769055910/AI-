# -*- coding: utf-8 -*-
"""
数据库连接模块 — pymysql + ai_picture
"""
import pymysql

DB_CONFIG = {
    'host': '127.0.0.1',
    'port': 3306,
    'user': 'root',
    'password': 'wa123456',
    'database': 'ai_picture',
    'charset': 'utf8mb4',
    'cursorclass': pymysql.cursors.DictCursor,
}


def get_db():
    """获取数据库连接"""
    return pymysql.connect(**DB_CONFIG)