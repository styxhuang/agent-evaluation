import asyncio
import json
import logging
import os
import re
import time
import uuid
from typing import Any, Dict, List

from bohrium import Bohrium
from dotenv import find_dotenv, load_dotenv
from google.adk import Runner
from google.adk.agents import RunConfig
from google.adk.agents.run_config import StreamingMode
from google.adk.artifacts import InMemoryArtifactService
from google.adk.sessions import InMemorySessionService
from google.genai import types

from agents.matmaster_agent.agent import root_agent
from .human_simulator import ConversationGoal, HumanSimulator
from ..utils import load_dataset_json

logger = logging.getLogger(__name__)

load_dotenv(find_dotenv(), override=True)
print(os.getenv('BOHRIUM_API_URL'))


async def _run_conversation(
    dataset_item: Dict[str, Any],
    max_turn_count: int,
    item_id: int,
    save_mode: str = 'w',
    label_key: str = '',
) -> Dict[str, Any]:
    """
    执行一次对话测试，并返回结果
    :param dataset_item: 单条测试数据
    :param max_turn_count: 最大对话轮次
    :param save_mode: 写文件模式 ("w" 覆盖 / "a" 追加)
    """
    if item_id is None:
        item_id = 0
    if not os.path.exists(f'logs/job_{item_id}'):
        os.makedirs(f'logs/job_{item_id}')

    session_service = InMemorySessionService()
    artifact_service = InMemoryArtifactService()
    session = await session_service.create_session(
        app_name='matmaster_agent',
        user_id='human_simulator_test',
    )

    logger.info(f"Test Session: {session.id}")

    runner = Runner(
        app_name='matmaster_agent',
        agent=root_agent,
        session_service=session_service,
        artifact_service=artifact_service,
    )

    simulator = HumanSimulator(max_turn_count=max_turn_count)

    # 场景初始化
    scenario = {
        'name': dataset_item['initial_question'],
        'goal': ConversationGoal(
            initial_question=dataset_item['initial_question'],
            expected_outcomes=dataset_item['expected_outcomes'],
            success_criteria=dataset_item['success_criteria'],
        ),
    }

    file_parts = []
    if 'file_urls' in dataset_item:
        for file_url in dataset_item['file_urls']:
            # with open(file_url, "rb") as f:
            #     file_bytes = f.read()
            file_part = types.Part.from_uri(
                file_uri=file_url, mime_type='application/pdf'
            )
            file_parts.append(file_part)

    print(f"\n{'=' * 20} 测试场景: {scenario['name']} {'=' * 20}")

    simulator.set_goal(scenario['goal'])
    initial_question = simulator.get_initial_question()

    print(f"🎯 对话目标: {initial_question}")
    print(f"📋 期望结果: {', '.join(scenario['goal'].expected_outcomes)}")
    print(f"✅ 成功标准: {', '.join(scenario['goal'].success_criteria)}")

    # 初始化结果
    eval_results = {
        'initial_question': initial_question,
        'expected_outcomes': scenario['goal'].expected_outcomes,
        'success_criteria': scenario['goal'].success_criteria,
    }
    for i in range(1, max_turn_count + 1):
        eval_results[f'agent_response_{i}'] = ''
        eval_results[f'user_response_{i}'] = ''

    # 对话循环
    turn_count = 0
    while turn_count < max_turn_count:
        if not os.path.exists(f"{label_key}/logs/job_{item_id}"):
            os.makedirs(f"{label_key}/logs/job_{item_id}")
        turn_count += 1
        print(f"\n🔄 第 {turn_count} 轮对话:")

        # 获取用户输入
        user_input = (
            initial_question if turn_count == 1 else simulator.get_last_user_response()
        )
        print(f"🧑 模拟用户: {user_input}")

        # 调用 agent
        try:
            if turn_count == 1 and file_parts != []:
                content = types.Content(
                    role='user', parts=file_parts + [types.Part(text=user_input)]
                )
            else:
                content = types.Content(
                    role='user', parts=[types.Part(text=user_input)]
                )
            agent_response = ''

            events = runner.run_async(
                user_id=session.user_id,
                session_id=session.id,
                new_message=content,
                run_config=RunConfig(streaming_mode=StreamingMode.SSE),
            )
            
            # ========================== #
            # 收集所有事件以供查看和后续处理  #
            # ========================== #
            events_list = []
            async for event in events:
                # 打印每个事件的内容，方便调试查看
                # print(f"DEBUG: Received event: {event}") 
                
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        if part.text:
                            agent_response += part.text
                        # 如果你想看 function_call 内容：
                        if part.function_call:
                            print(f"DEBUG: Function Call: {part.function_call}")
                            
                # 将事件转换为字典并保存
                events_list.append(dict(event))

            # 将事件保存到txt文件
            with open(
                f"{label_key}/logs/job_{item_id}/turn_{turn_count}.txt",
                'w',
                encoding='utf-8',
            ) as f:
                f.write(str(events_list))

        except asyncio.CancelledError:
            msg = '任务被取消，可能是超时或作用域取消导致'
            logger.error(msg)
            eval_results[f'agent_response_{turn_count}'] = msg
            raise
        except Exception as e:
            logger.error(f"获取agent响应失败: {e}")
            eval_results[f'agent_response_{turn_count}'] = str(e)
            raise e

        eval_results[f'agent_response_{turn_count}'] = agent_response
        print(f"🤖 ADK Agent: {agent_response}")

        # 提取 job_id
        job_jsons = re.findall(
            r'<bohrium-chat-msg>(.*?)</bohrium-chat-msg>', agent_response
        )
        job_ids: List[str] = []
        for job_json in job_jsons:
            try:
                job_json = json.loads(job_json)
                if 'eventData' in job_json and 'content' in job_json['eventData']:
                    content = job_json['eventData']['content']
                    if 'job_list' in content and 'job_id' in content['job_list']:
                        job_ids.append(content['job_list']['job_id'])
            except Exception as e:
                logger.error(f"提取job_id失败: {e}")

        # 查询 job 状态
        if job_ids:
            job_ids = list(set(job_ids))
            while True:
                time.sleep(10)
                all_finished = True
                for job_id in job_ids:
                    try:
                        bohrium_client = Bohrium(
                            base_url=os.getenv(
                                'BOHRIUM_API_URL',
                                'https://test.openapi.bohrium.dp.tech',
                            ),
                            access_key=os.getenv('MATERIALS_ACCESS_KEY'),
                            project_id=os.getenv('MATERIALS_PROJECT_ID'),
                        )
                        job_info = bohrium_client.job.detail(job_id)
                    except Exception as e:
                        import traceback

                        print(f"tracebackkkkkkkkkk, {traceback.print_exc()}")
                        logger.error(f"查询job状态失败: {e}")
                        all_finished = False
                        continue

                    logger.info(f"查询到job状态: {job_id} - 状态: {job_info['status']}")
                    if job_info['status'] not in [-1, 2]:
                        all_finished = False
                if all_finished:
                    break

            user_response, should_continue = simulator.get_bohr_results(
                agent_response, job_ids
            )
        else:
            user_response, should_continue = simulator.generate_response(agent_response)

        eval_results[f'user_response_{turn_count}'] = user_response
        print(f"🧑 模拟用户: {user_response}")

        if not should_continue:
            print(f"✅ 对话在第{turn_count}轮结束")
            break

    # 对话总结
    summary = simulator.get_conversation_summary()
    eval_results.update(
        {
            'total_turns': summary['total_turns'],
            'final_state': summary['final_state'],
            'duration_minutes': summary['duration_minutes'],
        }
    )

    print('\n📊 对话摘要:')
    print(f"   - 总轮次: {summary['total_turns']}")
    print(f"   - 最终状态: {summary['final_state']}")
    print(f"   - 耗时: {summary['duration_minutes']:.1f} 分钟")

    # 保存结果
    with open('evaluation_results.json', save_mode, encoding='utf-8') as f:
        json.dump(eval_results, f, indent=4, ensure_ascii=False)

    if summary['final_state'] == 'satisfied':
        print('✅ 测试通过: 对话成功完成')
    else:
        print('❌ 测试失败: 对话未成功完成')

    await runner.close()
    return eval_results


async def evaluation_threads_single_task(
    file_path: str,
    item_id: int,
    max_turn_count: int = 10,
    label_key: str = '',
    max_retries: int = 1,
    base_backoff: float = 5.0,
):
    """测试单个数据（带重试）"""
    print('=' * 80)
    print('🤖 与ADK Agent多轮对话测试')
    print('=' * 80)

    dataset_json = json.loads(load_dataset_json(file_path))
    dataset_item = dataset_json[item_id]
    time.sleep(10)  # 避免请求过于频繁

    attempt = 0
    while attempt < max_retries:
        try:
            result = await _run_conversation(
                dataset_item,
                max_turn_count,
                save_mode='a',
                item_id=item_id,
                label_key=label_key,
            )
            # 成功则跳出重试循环
            break
        except asyncio.CancelledError:
            # 取消应直接传播
            logger.error('任务被取消，停止重试')
            raise
        except Exception as e:
            attempt += 1
            logger.error(f"第 {attempt} 次执行失败: {e}")
            if attempt >= max_retries:
                logger.error('已达到最大重试次数，抛出异常')
                raise
            backoff = base_backoff * (2 ** (attempt - 1))
            print(f"⚠️ 第 {attempt} 次执行失败，{backoff} 秒后重试...")
            await asyncio.sleep(backoff)

    print('\n' + '=' * 80)
    print('🎉 单条多轮对话测试完成！')
    print('=' * 80)

    return result