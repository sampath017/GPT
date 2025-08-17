# GPT

## Training
uv run torchrun --standalone --nproc-per-node=2 src/main.py

## Training steps

step 5   | train_loss 7.83  | val_loss 7.81  | norm 0.75  | time 1297.64 ms
step 6   | train_loss 7.91  | val_loss 7.79  | norm 0.57  | time 1303.11 ms
step 7   | train_loss 7.80  | val_loss 7.86  | norm 0.86  | time 1300.03 ms
step 8   | train_loss 7.87  | val_loss 7.89  | norm 0.91  | time 1304.53 ms

# TO-UNDERSTAND
1. 
    why reverse weight sharing scheme is resulting in large loss ?
    step 0   | train_loss 472.79 | norm 95.00 | time 54479.77 ms | tok/sec 9.62K
    step 1   | train_loss 121.85 | norm 286.41 | time 48853.12 ms | tok/sec 10.73K
    step 2   | train_loss 106.30 | norm 127.80 | time 48945.54 ms | tok/sec 10.71K


# Previous A100 - 2 RUN
step 196 | train_loss 7.74  | norm 0.24  | time 1314.65 ms | tok/sec 398.81K
step 197 | train_loss 7.71  | norm 0.18  | time 1316.37 ms | tok/sec 398.28K
step 198 | train_loss 7.66  | norm 0.25  | time 1315.80 ms | tok/sec 398.46K
step 199 | train_loss 7.67  | norm 0.38  | time 1313.40 ms | tok/sec 399.18K


# latest run
step 11789 | train_loss 3.61  | norm 0.31  | time 2432.08 ms | tok/sec 215.57K
step 11790 | train_loss 3.58  | norm 0.33  | time 2314.52 ms | tok/sec 226.52K
step 11791 | train_loss 3.55  | norm 0.29  | time 2244.01 ms | tok/sec 233.64K
step 11792 | train_loss 3.60  | norm 0.33  | time 2281.14 ms | tok/sec 229.84K
step 11793 | train_loss 3.57  | norm 0.30  | time 2203.76 ms | tok/sec 237.91K
step 11794 | train_loss 3.56  | norm 0.32  | time 2235.46 ms | tok/sec 234.53K
step 11795 | train_loss 3.58  | norm 0.29  | time 2479.07 ms | tok/sec 211.49K
step 11796 | train_loss 3.53  | norm 0.30  | time 2429.73 ms | tok/sec 215.78K
step 11797 | train_loss 3.48  | norm 0.28  | time 2220.64 ms | tok/sec 236.10K
step 11798 | train_loss 3.55  | norm 0.32  | time 2312.74 ms | tok/sec 226.70K
step 11799 | train_loss 3.54  | norm 0.31  | time 2228.91 ms | tok/sec 235.22K
rank 0 sample 0: Hello, I'm a language model, and I know that it's very helpful -- one that's a powerful signal in the way that a person communicates with other
rank 0 sample 1: Hello, I'm a language model, but not a language, and not everybody is talking dialects, I prefer that. So the language does not translate into
rank 1 sample 0: Hello, I'm a language model, and you're a native speaker. These two kinds of metaphors use the same concept, so it works. The first is
rank 1 sample 1: Hello, I'm a language model, a way of working with people with mental health issues. A lot of people struggle with substance use and substance use.

rank 3 sample 0: Hello, I'm a language model, and I'm not an African-American.
It's quite easy. At a university degree, some of you get
rank 3 sample 1: Hello, I'm a language model, and so it's a major factor in my job. For example:
- I'm a language learner: You
rank 2 sample 0: Hello, I'm a language model, really good at speaking to someone. When you're using your computer, you can learn to speak, and use it,
rank 2 sample 1: Hello, I'm a language model, with a teacher in the classroom. That same year I began my classroom using the language model to help me make better decisions
HellaSwag accuracy: 2644/10042=0.2633
val loss 3.5919
step 11800 | train_loss 3.55  | norm 0.28  | time 19480.78 ms | tok/sec 26.91K
step 11801 | train_loss 3.49  | norm 0.30  | time 2058.30 ms | tok/sec 254.72K
step 11802 | train_loss 3.62  | norm 0.31  | time 2227.09 ms | tok/sec 235.41K
^CW0817 02:40:09.307000 17860 torch/distributed/elastic/agent/server/api.py:719] Received 2 death signal, shutting down workers
W0817 02:40:09.311000 17860 torch/distributed/elastic/multiprocessing/api.py:900] Sending process 17942 closing signal SIGINT
W0817 02:40:09.312000 17860 torch/distributed/elastic/multiprocessing/api.py:900] Sending process 17943 closing signal SIGINT
W0817 02:40:09.312000 17860 torch/distributed/elastic/multiprocessing/api.py:900] Sending process 17944 closing signal SIGINT
W0817 02:40:09.313000 17860 torch/distributed/elastic/multiprocessing/api.py:900] Sending process 17945 closing signal SIGINT
Stopping Run!
wandb:                                                                                
wandb: 
wandb: Run history:
wandb:    gpt2_hellaswag_acc ▁
wandb: gpt2_xl_hellaswag_acc ▁
wandb:         gradient_norm ▁▁▆▅██▄▄█▅▂▃▂▂▂▄▁▁▂▁▁▁▁▁▁▁▂▂▃▁▂▂▁▁▁▁▁▁▁▁
wandb:    hellaswag_accuracy ▄▃▄▃▃▄▃▃▁▁▃▃▃▃▄▄▅▆▆▆▇▆▇▇▇█▇▆▇▇▇█▇▇███▇█▇
wandb:                    lr ▂▄▇██████████▇▇▇▇▆▆▆▅▅▅▄▄▄▄▃▃▃▃▃▃▃▃▂▂▂▁▁
wandb:               tok/sec █▇██▇▆█▇▇▇▇▇▇▇▇█▇████▇██▁██████▇█████▇██
wandb:            train_loss █▇▇▇▆▅▅▅▅▅▄▃▃▃▂▂▂▂▂▂▂▂▁▂▁▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:            train_step ▁▁▁▁▁▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▅▅▅▆▆▆▇▇▇▇███
wandb:              val_loss ██▇▇▆▆▆▆▅▄▃▃▂▃▃▂▂▂▂▂▂▂▂▁▁▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb: 
wandb: Run summary:
wandb:    gpt2_hellaswag_acc 0.29655
wandb: gpt2_xl_hellaswag_acc 0.48925
wandb:         gradient_norm 0.30936
wandb:    hellaswag_accuracy 0.26329
wandb:                    lr 0.00019
wandb:               tok/sec 235414.1399
wandb:            train_loss 3.61514
wandb:            train_step 11802
wandb:              val_loss 3.59187
wandb: 
wandb: 🚀 View run upbeat-meadow-81 at: https://wandb.ai/sampath017/GPT3-124M/runs/4qvedfcy
wandb: ⭐️ View project at: https://wandb.ai/sampath017/GPT3-124M
wandb: Synced 5 W&B file(s), 0 media file(s), 2 artifact file(s) and 0 other file(s)
wandb: Find logs at: ./logs/wandb/run-20250816_185036-4qvedfcy/logs
W0817 02:40:39.314000 17860 torch/distributed/elastic/multiprocessing/api.py:919] Unable to shutdown process 17943 via 2, forcefully exiting via 9
Traceback (most recent call last):
  File "/workspace/GPT/.venv/bin/torchrun", line 10, in <module>
    sys.exit(main())
             ~~~~^^
  File "/workspace/GPT/.venv/lib/python3.13/site-packages/torch/distributed/elastic/multiprocessing/errors/__init__.py", line 355, in wrapper
    return f(*args, **kwargs)
  File "/workspace/GPT/.venv/lib/python3.13/site-packages/torch/distributed/run.py", line 892, in main
    run(args)
    ~~~^^^^^^
  File "/workspace/GPT/.venv/lib/python3.13/site-packages/torch/distributed/run.py", line 883, in run
    elastic_launch(
    ~~~~~~~~~~~~~~~
        config=config,
        ~~~~~~~~~~~~~~
        entrypoint=cmd,
        ~~~~~~~~~~~~~~~
    )(*cmd_args)
    ~^^^^^^^^^^^
  File "/workspace/GPT/.venv/lib/python3.13/site-packages/torch/distributed/launcher/api.py", line 139, in __call__
    return launch_agent(self._config, self._entrypoint, list(args))
  File "/workspace/GPT/.venv/lib/python3.13/site-packages/torch/distributed/launcher/api.py", line 261, in launch_agent
    result = agent.run()
  File "/workspace/GPT/.venv/lib/python3.13/site-packages/torch/distributed/elastic/metrics/api.py", line 138, in wrapper
    result = f(*args, **kwargs)
  File "/workspace/GPT/.venv/lib/python3.13/site-packages/torch/distributed/elastic/agent/server/api.py", line 711, in run
    result = self._invoke_run(role)
  File "/workspace/GPT/.venv/lib/python3.13/site-packages/torch/distributed/elastic/agent/server/api.py", line 870, in _invoke_run
    time.sleep(monitor_interval)
    ~~~~~~~~~~^^^^^^^^^^^^^^^^^^
  File "/workspace/GPT/.venv/lib/python3.13/site-packages/torch/distributed/elastic/multiprocessing/api.py", line 84, in _terminate_process_handler
    raise SignalException(f"Process {os.getpid()} got signal: {sigval}", sigval=sigval)
torch.distributed.elastic.multiprocessing.api.SignalException: Process 17860 got signal: 2


## 
1. Test overfit batch
2. load checkpoint  