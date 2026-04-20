# -*- coding: utf-8 -*-
"""
简单聊天机器人 - 基于本地模型的命令行对话
"""

import sys
import json
import os
from pathlib import Path
from datetime import datetime

# 添加项目根目录到Python路径
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))


def execute_action(action_name: str, params: dict, reminders_file: Path):
    """执行具体操作"""
    if action_name == "create_reminder":
        manage_reminders("create_reminder", params, reminders_file)
    elif action_name == "delete_reminder":
        manage_reminders("delete_reminder", params, reminders_file)
    elif action_name == "list_reminders":
        manage_reminders("list_reminders", params, reminders_file)
    else:
        # 其他操作只是显示执行信息
        print(f"   ✅ {action_name} 操作已计划")


def manage_reminders(action: str, params: dict, reminders_file: Path):
    """管理提醒事项"""
    try:
        # 读取现有提醒
        if reminders_file.exists():
            reminders = json.loads(reminders_file.read_text(encoding='utf-8'))
        else:
            reminders = []
        
        if action == "create_reminder":
            # 创建新提醒
            reminder_id = f"reminder_{len(reminders) + 1}_{int(datetime.now().timestamp())}"
            new_reminder = {
                "id": reminder_id,
                "time": params.get("time", ""),
                "content": params.get("content", ""),
                "repeat": params.get("repeat", "once"),
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "status": "active"
            }
            reminders.append(new_reminder)
            reminders_file.write_text(json.dumps(reminders, ensure_ascii=False, indent=2), encoding='utf-8')
            print(f"   ✅ 提醒已保存 (ID: {reminder_id})")
            return new_reminder
            
        elif action == "delete_reminder":
            # 删除提醒
            reminder_id = params.get("id", "")
            reminders = [r for r in reminders if r.get("id") != reminder_id]
            reminders_file.write_text(json.dumps(reminders, ensure_ascii=False, indent=2), encoding='utf-8')
            print(f"   ✅ 提醒已删除 (ID: {reminder_id})")
            
        elif action == "list_reminders":
            # 列出所有提醒
            active_reminders = [r for r in reminders if r.get("status") == "active"]
            if active_reminders:
                print(f"\n📋 当前提醒列表:")
                for r in active_reminders:
                    print(f"   - [{r['id']}] {r['time']} - {r['content']} ({r['repeat']})")
            else:
                print(f"\n📋 当前没有活跃的提醒")
            return active_reminders
            
    except Exception as e:
        print(f"   ❌ 操作失败: {e}")


def run():
    """主入口函数"""
    print("=" * 60)
    print("🤖 简单聊天机器人")
    print("   输入 'quit' 或 'exit' 退出")
    print("=" * 60 + "\n")
    
    # 初始化模型
    from inference.model_manager import ModelManager
    
    # 简化配置
    class SimpleConfig:
        model_type = "gguf"
        gguf_path = "./models/qwen2.5-3b-instruct-q4_k_m.gguf"
        download_url = ""
        
        class Inference:
            n_ctx = 2048
            n_threads = 4
            n_gpu_layers = 0
            temperature = 0.7
            top_p = 0.9
            max_tokens = 512
        
        inference = Inference()
    
    config = SimpleConfig()
    model_manager = ModelManager(config)
    
    try:
        model_manager.initialize()
        print("✅ 模型加载成功!\n")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    # 初始化数据存储
    reminders_file = PROJECT_ROOT / "data" / "reminders.json"
    reminders_file.parent.mkdir(exist_ok=True)
    if not reminders_file.exists():
        reminders_file.write_text(json.dumps([], ensure_ascii=False, indent=2))
    
    # 聊天循环
    messages = []
    system_prompt = """你是一个智能家庭助手，具备任务规划和自动执行能力。

当用户提出需求时，你需要：
1. 分析用户意图
2. 规划完成任务所需的具体操作步骤
3. 以标准化的JSON格式输出操作计划

## 输出格式要求

你必须严格按照以下JSON格式输出：

```json
{
  "intent": "意图类型",
  "thinking": "你的思考过程，分析用户需求和可能的解决方案",
  "actions": [
    {
      "action": "操作名称",
      "params": {"参数名": "参数值"},
      "description": "这个操作的说明"
    }
  ],
  "response": "给用户的自然语言回复"
}
```

## 支持的意图类型和操作

### 1. entertainment（娱乐场景）
- adjust_lights: 调节灯光 {"brightness": 0-100, "color": "warm/cool/white"}
- control_curtains: 控制窗帘 {"action": "open/close", "position": 0-100}
- control_tv: 控制电视 {"action": "on/off/volume/channel"}
- set_mode: 设置场景模式 {"mode": "movie/music/reading/sleep"}

### 2. reminder（提醒管理）
- create_reminder: 创建提醒 {"time": "HH:MM", "content": "提醒内容", "repeat": "once/daily/weekly"}
- delete_reminder: 删除提醒 {"id": "提醒ID"}
- list_reminders: 列出所有提醒 {}

### 3. health（健康管理）
- record_health: 记录健康数据 {"type": "blood_pressure/blood_sugar/heart_rate", "value": "数值"}
- query_health: 查询健康记录 {"type": "数据类型", "period": "today/week/month"}

### 4. smart_home（智能家居控制）
- control_device: 控制设备 {"device": "设备名", "action": "on/off/set", "value": "设定值"}
- query_status: 查询状态 {"device": "设备名"}

### 5. chat（普通对话）
- 无需actions，直接回复即可

## 示例

用户：想看电影
输出：
```json
{
  "intent": "entertainment",
  "thinking": "用户想看电影，需要营造影院氛围：调暗灯光、关闭窗帘、打开电视",
  "actions": [
    {"action": "adjust_lights", "params": {"brightness": 20, "color": "warm"}, "description": "调暗灯光到20%，暖色调"},
    {"action": "control_curtains", "params": {"action": "close", "position": 0}, "description": "完全关闭窗帘"},
    {"action": "control_tv", "params": {"action": "on"}, "description": "打开电视"},
    {"action": "set_mode", "params": {"mode": "movie"}, "description": "切换到电影模式"}
  ],
  "response": "已为您准备好观影环境：灯光调暗、窗帘关闭、电视已开启。祝您观影愉快！🎬"
}
```

用户：提醒我晚上8点吃药
输出：
```json
{
  "intent": "reminder",
  "thinking": "用户需要在晚上8点吃药，创建一个一次性提醒",
  "actions": [
    {"action": "create_reminder", "params": {"time": "20:00", "content": "吃药", "repeat": "once"}, "description": "创建今晚8点的吃药提醒"}
  ],
  "response": "已设置提醒：今晚8点吃药⏰ 到时我会提醒您。"
}
```

用户：你好
输出：
```json
{
  "intent": "chat",
  "thinking": "用户只是打招呼，无需执行任何操作",
  "actions": [],
  "response": "您好！我是您的智能家庭助手，可以帮您控制家居设备、设置提醒、管理健康数据等。有什么可以帮您的吗？😊"
}
```

## 重要规则

1. 必须输出有效的JSON格式
2. thinking字段要体现你的推理过程
3. actions数组可以为空（普通对话时）
4. response字段是给用户的友好回复
5. 参数值要具体明确
6. 如果用户需求不清晰，可以在response中询问澄清"""
    
    while True:
        try:
            user_input = input("👤 你: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', '退出']:
                print("\n再见! 👋")
                break
            
            # 构建消息历史
            messages.append({"role": "user", "content": user_input})
            
            # 构建完整prompt
            full_messages = [{"role": "system", "content": system_prompt}] + messages
            
            # 调用模型
            response = model_manager._llm.create_chat_completion(
                messages=full_messages,
                temperature=config.inference.temperature,
                top_p=config.inference.top_p,
                max_tokens=config.inference.max_tokens
            )
            
            # 提取回复
            if response and 'choices' in response and len(response['choices']) > 0:
                bot_reply = response['choices'][0]['message']['content'].strip()
                
                # 尝试解析JSON输出
                try:
                    # 提取JSON部分（可能在代码块中）
                    json_str = bot_reply
                    if '```json' in bot_reply:
                        json_str = bot_reply.split('```json')[1].split('```')[0].strip()
                    elif '```' in bot_reply:
                        json_str = bot_reply.split('```')[1].split('```')[0].strip()
                    
                    result = json.loads(json_str)
                    
                    # 显示思考过程
                    if 'thinking' in result:
                        print(f"\n💭 思考: {result['thinking']}")
                    
                    # 执行操作
                    if 'actions' in result and result['actions']:
                        print(f"\n🔧 执行操作:")
                        for i, action in enumerate(result['actions'], 1):
                            action_name = action.get('action', 'unknown')
                            params = action.get('params', {})
                            desc = action.get('description', '')
                            print(f"   {i}. {action_name}: {desc}")
                            print(f"      参数: {json.dumps(params, ensure_ascii=False)}")
                            # 执行操作
                            execute_action(action_name, params, reminders_file)
                    
                    # 显示回复
                    if 'response' in result:
                        print(f"\n🤖 助手: {result['response']}\n")
                    
                    # 保存到历史（保存完整JSON）
                    messages.append({"role": "assistant", "content": bot_reply})
                    
                except (json.JSONDecodeError, KeyError) as e:
                    # 如果不是JSON格式，直接显示
                    print(f"\n🤖 机器人: {bot_reply}\n")
                    messages.append({"role": "assistant", "content": bot_reply})
                
                # 限制历史长度
                if len(messages) > 20:
                    messages = messages[-20:]
                
                print(f"\n🤖 机器人: {bot_reply}\n")
            else:
                print("\n🤖 机器人: 抱歉，我没有生成回复。\n")
                
        except KeyboardInterrupt:
            print("\n\n再见! 👋")
            break
        except Exception as e:
            print(f"\n❌ 错误: {e}\n")


if __name__ == "__main__":
    run()
