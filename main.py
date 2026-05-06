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

# 导入智能提醒管理器
from reminder_manager import ReminderManager
# 导入多Agent系统
from multi_agent_system import MultiAgentSystem


def execute_action(action_name: str, params: dict, reminder_manager: ReminderManager):
    """执行具体操作"""
    if action_name == "create_reminder":
        try:
            time_str = params.get("time", "")
            content = params.get("content", "")
            repeat = params.get("repeat", "once")
            
            reminder = reminder_manager.create_reminder(time_str, content, repeat)
            print(f"   ✅ 提醒已创建")
            print(f"      ID: {reminder['id']}")
            print(f"      时间: {reminder['time_display']}")
            print(f"      内容: {reminder['content']}")
            return reminder
        except ValueError as e:
            print(f"   ❌ 创建失败: {e}")
            return None
    elif action_name == "delete_reminder":
        reminder_id = params.get("id", "")
        success = reminder_manager.delete_reminder(reminder_id)
        if success:
            print(f"   ✅ 提醒已删除 (ID: {reminder_id})")
        else:
            print(f"   ❌ 未找到提醒 (ID: {reminder_id})")
    elif action_name == "list_reminders":
        reminders = reminder_manager.list_reminders()
        if reminders:
            print(f"\n📋 当前提醒列表:")
            for r in reminders:
                print(f"   - [{r['id']}] {r['time_display']} - {r['content']} ({r['repeat']})")
        else:
            print(f"\n📋 当前没有活跃的提醒")
        return reminders
    elif action_name == "control_device":
        # 智能家居设备控制 - 需要安全评估
        device = params.get("device", "")
        action = params.get("action", "")
        value = params.get("value", None)
        
        print(f"   📝 设备: {device}")
        print(f"   📝 操作: {action}")
        print(f"   📝 参数: {value}")
        
        # 执行安全评估
        assessment = SafetyAssessment.assess_operation(device, action, params)
        
        # 显示安全报告
        safety_report = SafetyAssessment.generate_safety_report(assessment)
        if safety_report:
            print(safety_report)
        
        # 如果操作不安全，拒绝执行
        if not assessment["is_safe"]:
            print(f"   ❌ 操作被阻止：存在安全隐患")
            if assessment["auto_adjusted"]:
                print(f"   💡 建议的安全参数: {assessment['adjusted_params']}")
            return None
        
        # 如果需要确认，提示用户
        if assessment["should_confirm"]:
            print(f"   ⚡ 此操作需要您的确认")
            # 在实际应用中，这里应该等待用户确认
            # 为了演示，我们继续执行但给出警告
        
        # 如果自动调整了参数，使用调整后的值
        if assessment["auto_adjusted"]:
            print(f"   ✅ 已使用安全参数执行")
            final_params = assessment["adjusted_params"]
        else:
            final_params = params
        
        print(f"   ✅ 设备控制命令已发送")
        return final_params
    else:
        # 其他操作只是显示执行信息
        print(f"   ✅ {action_name} 操作已计划")





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
            n_ctx = 32768
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
    
    # 初始化智能提醒管理器
    reminders_file = PROJECT_ROOT / "data" / "reminders.json"
    reminder_manager = ReminderManager(reminders_file)
    
    # 初始化多Agent系统
    agent_system = MultiAgentSystem()
    
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
- create_reminder: 创建提醒 {"time": "时间表达式", "content": "提醒内容", "repeat": "once/daily/weekly"}
  * 时间表达式支持：
    - 绝对时间："20:00", "晚上8点", "明天上午9:30"
    - 相对时间："1小时后", "30分钟后", "明天"
    - 混合表达："今天晚上8点", "后天早上7点"
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
  "thinking": "用户需要在晚上8点吃药。当前时间是{current_time}，需要解析'晚上8点'为具体时间20:00。创建一个一次性提醒。",
  "actions": [
    {"action": "create_reminder", "params": {"time": "晚上8点", "content": "吃药", "repeat": "once"}, "description": "创建今晚8点的吃药提醒"}
  ],
  "response": "已设置提醒：今晚8点吃药⏰ 到时我会提醒您。"
}
```

用户：1小时后提醒我开会
输出：
```json
{
  "intent": "reminder",
  "thinking": "用户需要1小时后开会。当前时间是{current_time}，1小时后是{target_time}。创建相对时间提醒。",
  "actions": [
    {"action": "create_reminder", "params": {"time": "1小时后", "content": "开会", "repeat": "once"}, "description": "创建1小时后的开会提醒"}
  ],
  "response": "已设置提醒：1小时后开会⏰"
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
6. 如果用户需求不清晰，可以在response中询问澄清

## ⚠️ 安全规范（非常重要）

在控制智能家居设备时，你必须考虑用户安全：

### 热水器温度控制
- **安全范围**：30-60°C
- **推荐温度**：40-45°C（洗澡）
- **危险温度**：超过60°C可能导致烫伤
- **最高限制**：75°C
- **特殊人群**：老人、儿童使用时应设置更低温度（38-42°C）
- ❌ 禁止设置过高温度（如“调到最高”、“加热到最大”）
- ✅ 应该建议安全温度：“已设置为45°C，这是舒适的洗澡温度”

### 燃气灶使用
- **火力等级**：1-5档
- **安全提醒**：使用时保持通风，离开前关闭
- ❌ 禁止无人看管时使用高火力
- ✅ 应该提醒：“已开启燃气灶，请注意安全”

### 空调温度
- **舒适范围**：20-28°C
- **推荐温度**：夏季26°C，冬季22°C
- **健康提醒**：避免温度过低或过高
- ✅ 应该建议合理温度

### 通用原则
1. **优先考虑安全**：当用户请求可能危险的操作时，应该拒绝或调整到安全范围
2. **提供建议**：解释为什么某个操作不安全，并给出更安全的替代方案
3. **特殊关怀**：考虑到老人、儿童等特殊群体的需求
4. **风险告知**：对于中等风险操作，应该告知风险并建议确认

### 示例对比

❌ 错误的响应：
```json
{
  "thinking": "用户想要调高热水器温度，设置为70度",
  "actions": [{"action": "control_device", "params": {"device": "热水器", "value": 70}}],
  "response": "已将热水器调到70度"
}
```

✅ 正确的响应：
```json
{
  "thinking": "用户想要调高热水器温度。70°C过高，有烫伤风险。应该设置为安全的45°C，并告知用户原因。",
  "actions": [{"action": "control_device", "params": {"device": "热水器", "value": 45}, "description": "设置为安全温度45°C"}],
  "response": "已将热水器设置为45°C。这个温度适合洗澡，既舒适又安全。如果需要更高温度，请注意防烫伤。"
}
```"""
    
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
                        print(f"\n💭 AI思考: {result['thinking']}")
                    
                    # 检查是否有设备控制操作
                    has_device_control = False
                    if 'actions' in result and result['actions']:
                        for action in result['actions']:
                            if action.get('action') == 'control_device':
                                has_device_control = True
                                break
                    
                    # 如果有设备控制，使用多Agent系统处理
                    if has_device_control:
                        print(f"\n🔄 检测到设备控制，启动多Agent协作流程...")
                        agent_result = agent_system.process_request(user_input)
                        
                        # 显示Agent协作结果
                        print(f"\n💭 MainAgent思考: {agent_result.get('thinking', '')}")
                        
                        if agent_result.get("actions"):
                            print(f"\n🔧 Agent执行的操作:")
                            for i, action in enumerate(agent_result['actions'], 1):
                                print(f"   {i}. {action['action']}: {action.get('description', '')}")
                        
                        # 显示安全评估
                        if agent_result.get("safety_assessment"):
                            assessment = agent_result["safety_assessment"]
                            if not assessment['is_safe'] or assessment['warnings']:
                                print(f"\n🛡️  安全评估:")
                                if assessment['warnings']:
                                    for w in assessment['warnings']:
                                        print(f"   ⚠️  {w}")
                                if assessment.get('auto_adjusted'):
                                    print(f"   ✅ 已自动调整参数: {assessment['adjusted_params']}")
                        
                        # 显示最终回复
                        if agent_result.get('response'):
                            print(f"\n🤖 助手: {agent_result['response']}\n")
                    else:
                        # 非设备控制操作，直接执行
                        if 'actions' in result and result['actions']:
                            print(f"\n🔧 执行操作:")
                            for i, action in enumerate(result['actions'], 1):
                                action_name = action.get('action', 'unknown')
                                params = action.get('params', {})
                                desc = action.get('description', '')
                                print(f"   {i}. {action_name}: {desc}")
                                print(f"      参数: {json.dumps(params, ensure_ascii=False)}")
                                # 执行操作
                                execute_action(action_name, params, reminder_manager)
                        
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
