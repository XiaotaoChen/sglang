from evalscope import TaskConfig, run_task
from evalscope.constants import EvalType

task_cfg = TaskConfig(
    model='/mnt/xtchen/model/DeepSeek-R1-W4AFP8',   # 模型名称 (需要与部署时的模型名称一致)
    api_url='http://127.0.0.1:8000/v1',  # 推理服务地址
    api_key='EMPTY',
    eval_type=EvalType.SERVICE,   # 评测类型，SERVICE表示评测推理服务
    datasets=[
        'data_collection',  # 数据集名称(固定为data_collection表示使用混合数据集)
    ],
    dataset_args={
        'data_collection': {
            'dataset_id': 'modelscope/R1-Distill-Math-Test'  # 数据集ID 或 数据集本地路径
        }
    },
    eval_batch_size=8,       # 发送请求的并发数
    generation_config={       # 模型推理配置
        'max_tokens': 20000,  # 最大生成token数，建议设置为较大值避免输出截断
        'temperature': 0.6,   # 采样温度 (deepseek 报告推荐值)
        'top_p': 0.95,        # top-p采样 (deepseek 报告推荐值)
        'n': 1                # 每个请求产生的回复数量 (注意 lmdeploy 目前只支持 n=1)
    },
    stream=True               # 是否使用流式请求，推荐设置为True防止请求超时
)

run_task(task_cfg=task_cfg)
