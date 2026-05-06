# -*- coding: utf-8 -*-
"""
智能提醒管理模块
支持实时时间获取和自然语言时间解析
"""

import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List


class TimeParser:
    """自然语言时间解析器"""
    
    def __init__(self):
        self.now = datetime.now()
    
    def parse_time_expression(self, time_str: str) -> Optional[datetime]:
        """
        解析自然语言时间表达式
        支持：
        - 绝对时间："20:00", "晚上8点", "明天上午9:30"
        - 相对时间："1小时后", "30分钟后", "明天这个时候"
        - 混合表达："今天晚上8点", "后天早上7点"
        """
        if not time_str or not time_str.strip():
            return None
        
        time_str = time_str.strip()
        
        # 尝试解析各种时间格式
        result = self._parse_relative_time(time_str)
        if result:
            return result
        
        result = self._parse_absolute_time(time_str)
        if result:
            return result
        
        result = self._parse_mixed_time(time_str)
        if result:
            return result
        
        return None
    
    def _parse_relative_time(self, time_str: str) -> Optional[datetime]:
        """解析相对时间，如'1小时后'、'30分钟后'、'明天'"""
        
        # X小时后
        match = re.search(r'(\d+)\s*小时(?:后|之后)', time_str)
        if match:
            hours = int(match.group(1))
            return self.now + timedelta(hours=hours)
        
        # X分钟后
        match = re.search(r'(\d+)\s*分钟(?:后|之后)', time_str)
        if match:
            minutes = int(match.group(1))
            return self.now + timedelta(minutes=minutes)
        
        # X天后
        match = re.search(r'(\d+)\s*天(?:后|之后)', time_str)
        if match:
            days = int(match.group(1))
            return self.now + timedelta(days=days)
        
        # 明天/后天/大后天 - 只处理纯日期，不包含具体时间的情况
        if '明天' in time_str or '明日' in time_str:
            # 检查是否包含具体时间表达（如"9点"、"晚上8点"等）
            has_specific_time = re.search(r'\d{1,2}\s*点', time_str) or re.search(r'\d{1,2}[:：]\d{2}', time_str)
            if not has_specific_time:
                # 纯"明天"，返回明天同一时间
                target_date = self.now + timedelta(days=1)
                return target_date
            # 如果有具体时间，交给 _parse_mixed_time 处理
            return None
        
        if '后天' in time_str:
            # 检查是否包含具体时间表达
            has_specific_time = re.search(r'\d{1,2}\s*点', time_str) or re.search(r'\d{1,2}[:：]\d{2}', time_str)
            if not has_specific_time:
                # 纯"后天"，返回后天同一时间
                target_date = self.now + timedelta(days=2)
                return target_date
            # 如果有具体时间，交给 _parse_mixed_time 处理
            return None
        
        # 今天稍后/待会/一会
        if any(word in time_str for word in ['待会', '一会', '稍后']):
            return self.now + timedelta(minutes=30)
        
        return None
    
    def _parse_absolute_time(self, time_str: str) -> Optional[datetime]:
        """解析绝对时间，如'20:00'、'晚上8点'"""
        
        # 如果包含“明天”、“后天”等词，不应该在这里处理
        if any(word in time_str for word in ['明天', '明日', '后天', '今天']):
            return None
        
        # 标准时间格式 HH:MM
        match = re.match(r'^(\d{1,2})[:：](\d{2})$', time_str)
        if match:
            hour = int(match.group(1))
            minute = int(match.group(2))
            if 0 <= hour <= 23 and 0 <= minute <= 59:
                target = self.now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                # 如果时间已经过去，设为明天
                if target < self.now:
                    target += timedelta(days=1)
                return target
        
        # 中文时间表达：X点X分
        match = re.search(r'(\d{1,2})点(?:(\d{1,2})分)?', time_str)
        if match:
            hour = int(match.group(1))
            minute = int(match.group(2)) if match.group(2) else 0
            
            # 处理晚上/下午的12小时制转换
            if any(word in time_str for word in ['晚上', '晚上', '傍晚']):
                if hour < 12:
                    hour += 12
            elif any(word in time_str for word in ['下午']):
                if hour < 12:
                    hour += 12
            elif any(word in time_str for word in ['早上', '上午', '早晨']):
                if hour == 12:
                    hour = 0
            
            if 0 <= hour <= 23 and 0 <= minute <= 59:
                target = self.now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                # 如果是早上/上午且时间已过，可能是明天
                if target < self.now and any(word in time_str for word in ['早上', '上午', '早晨']):
                    target += timedelta(days=1)
                # 如果是晚上且时间已过，设为明天
                elif target < self.now and any(word in time_str for word in ['晚上', '下午']):
                    target += timedelta(days=1)
                # 如果没有明确时段且时间已过，设为明天
                elif target < self.now:
                    target += timedelta(days=1)
                return target
        
        return None
    
    def _parse_mixed_time(self, time_str: str) -> Optional[datetime]:
        """解析混合时间表达，如'明天晚上8点'、'后天早上7:30'、'明天9点'"""
        
        # 提取日期部分
        days_offset = 0
        if '明天' in time_str or '明日' in time_str:
            days_offset = 1
        elif '后天' in time_str:
            days_offset = 2
        
        if days_offset > 0:
            target_date = self.now + timedelta(days=days_offset)
            
            # 提取时间部分 - 改进正则，支持多种格式
            # 匹配：9点、09点、9:30、09:30、9时30分等
            time_match = re.search(r'(\d{1,2})\s*[:：]\s*(\d{2})', time_str)  # HH:MM 格式
            if not time_match:
                time_match = re.search(r'(\d{1,2})\s*点\s*(\d{1,2})\s*分', time_str)  # X点X分
            if not time_match:
                time_match = re.search(r'(\d{1,2})\s*点', time_str)  # X点
            
            if time_match:
                hour = int(time_match.group(1))
                minute = int(time_match.group(2)) if time_match.lastindex >= 2 and time_match.group(2) else 0
                
                # 时段判断
                if any(word in time_str for word in ['晚上', '傍晚']):
                    if hour < 12:
                        hour += 12
                elif any(word in time_str for word in ['下午']):
                    if hour < 12:
                        hour += 12
                elif any(word in time_str for word in ['早上', '上午', '早晨']):
                    if hour == 12:
                        hour = 0
                
                if 0 <= hour <= 23 and 0 <= minute <= 59:
                    return target_date.replace(hour=hour, minute=minute, second=0, microsecond=0)
            
            # 如果无法解析具体时间，返回None而不是默认时间
            return None
        
        return None
    
    def format_datetime(self, dt: datetime) -> str:
        """格式化日期时间为友好显示"""
        now = datetime.now()
        # 获取今天的日期（忽略时间）
        today = now.replace(hour=0, minute=0, second=0, microsecond=0)
        target_date = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        diff_days = (target_date - today).days
        
        if diff_days == 0:
            # 今天
            return f"今天 {dt.strftime('%H:%M')}"
        elif diff_days == 1:
            # 明天
            return f"明天 {dt.strftime('%H:%M')}"
        elif diff_days == 2:
            # 后天
            return f"后天 {dt.strftime('%H:%M')}"
        else:
            # 其他日期
            return dt.strftime('%Y-%m-%d %H:%M')


class ReminderManager:
    """提醒管理器"""
    
    def __init__(self, data_file: Path):
        self.data_file = data_file
        self.time_parser = TimeParser()
        self._ensure_data_file()
    
    def _ensure_data_file(self):
        """确保数据文件存在"""
        if not self.data_file.exists():
            self.data_file.parent.mkdir(parents=True, exist_ok=True)
            self.data_file.write_text(json.dumps([], ensure_ascii=False, indent=2), encoding='utf-8')
    
    def _load_reminders(self) -> List[Dict]:
        """加载提醒列表"""
        try:
            if self.data_file.exists():
                content = self.data_file.read_text(encoding='utf-8')
                return json.loads(content)
        except Exception as e:
            print(f"⚠️ 加载提醒失败: {e}")
        return []
    
    def _save_reminders(self, reminders: List[Dict]):
        """保存提醒列表"""
        try:
            self.data_file.write_text(
                json.dumps(reminders, ensure_ascii=False, indent=2),
                encoding='utf-8'
            )
        except Exception as e:
            print(f"❌ 保存提醒失败: {e}")
    
    def create_reminder(self, time_str: str, content: str, repeat: str = "once") -> Dict:
        """
        创建提醒
        :param time_str: 时间字符串（支持自然语言）
        :param content: 提醒内容
        :param repeat: 重复类型 (once/daily/weekly)
        :return: 创建的提醒对象
        """
        # 实时获取当前时间并解析
        target_time = self.time_parser.parse_time_expression(time_str)
        
        if not target_time:
            raise ValueError(f"无法解析时间表达式: '{time_str}'")
        
        # 生成唯一ID
        reminders = self._load_reminders()
        reminder_id = f"reminder_{len(reminders) + 1}_{int(datetime.now().timestamp())}"
        
        # 创建提醒对象
        new_reminder = {
            "id": reminder_id,
            "time": target_time.strftime("%Y-%m-%d %H:%M:%S"),
            "time_display": self.time_parser.format_datetime(target_time),
            "content": content,
            "repeat": repeat,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "active"
        }
        
        reminders.append(new_reminder)
        self._save_reminders(reminders)
        
        return new_reminder
    
    def delete_reminder(self, reminder_id: str) -> bool:
        """删除提醒"""
        reminders = self._load_reminders()
        original_count = len(reminders)
        reminders = [r for r in reminders if r.get("id") != reminder_id]
        
        if len(reminders) < original_count:
            self._save_reminders(reminders)
            return True
        return False
    
    def list_reminders(self, active_only: bool = True) -> List[Dict]:
        """列出提醒"""
        reminders = self._load_reminders()
        if active_only:
            reminders = [r for r in reminders if r.get("status") == "active"]
        
        # 按时间排序
        reminders.sort(key=lambda x: x.get("time", ""))
        return reminders
    
    def check_due_reminders(self) -> List[Dict]:
        """检查到期的提醒"""
        now = datetime.now()
        reminders = self._load_reminders()
        due_reminders = []
        
        for reminder in reminders:
            if reminder.get("status") != "active":
                continue
            
            try:
                reminder_time = datetime.strptime(reminder["time"], "%Y-%m-%d %H:%M:%S")
                # 允许1分钟的误差
                if abs((now - reminder_time).total_seconds()) <= 60:
                    due_reminders.append(reminder)
            except (ValueError, KeyError):
                continue
        
        return due_reminders
    
    def get_reminder_stats(self) -> Dict:
        """获取提醒统计信息"""
        reminders = self._load_reminders()
        active = [r for r in reminders if r.get("status") == "active"]
        
        stats = {
            "total": len(reminders),
            "active": len(active),
            "today": 0,
            "tomorrow": 0
        }
        
        now = datetime.now()
        today_end = now.replace(hour=23, minute=59, second=59)
        tomorrow_end = (now + timedelta(days=1)).replace(hour=23, minute=59, second=59)
        
        for reminder in active:
            try:
                reminder_time = datetime.strptime(reminder["time"], "%Y-%m-%d %H:%M:%S")
                if reminder_time <= today_end:
                    stats["today"] += 1
                elif reminder_time <= tomorrow_end:
                    stats["tomorrow"] += 1
            except (ValueError, KeyError):
                continue
        
        return stats
