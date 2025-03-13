
import openai
from time import sleep

from openai.error import RateLimitError, APIConnectionError
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import TerminalFormatter
from utils import load_prompt, DynamicObservation, IterableDynamicObservation
import time
from LLM_cache import DiskCache

'''
用于让 LLM（大语言模型）控制机器人
采用 "Code as Policies" 思想（用代码直接指导机器人行动）
基于 GPT-4 API，让 LMP 生成 Python 代码
'''
class LMP:
    """Language Model Program (LMP), adopted from Code as Policies."""
    def __init__(self, name, cfg, fixed_vars, variable_vars, debug=False, env='rlbench'):
        self._name = name
        self._cfg = cfg
        self._debug = debug
        self._base_prompt = load_prompt(f"{env}/{self._cfg['prompt_fname']}.txt") 
        #  LLM 通过这个 prompt 生成 Python 代码
        self._stop_tokens = list(self._cfg['stop'])
        self._fixed_vars = fixed_vars
        self._variable_vars = variable_vars
        self.exec_hist = ''
        # 记录 LMP 生成的执行历史
        # 可能用于调试和优化 LLM 生成的代码
        self._context = None
        self._cache = DiskCache(load_cache=self._cfg['load_cache'])
        # DiskCache 可能是一个缓存机制
        # 避免 LLM 反复调用 OpenAI API，降低 API 调用成本
        # 存储 LMP 生成的代码

    #清空执行历史
    def clear_exec_hist(self):
        self.exec_hist = ''

    #
    def build_prompt(self, query):

        # 将所有环境 API 的名称导入进来
        if len(self._variable_vars) > 0:
            variable_vars_imports_str = f"from utils import {', '.join(self._variable_vars.keys())}"
        else:
            variable_vars_imports_str = ''
        prompt = self._base_prompt.replace('{variable_vars_imports}', variable_vars_imports_str)

        # 如果配置要求维护会话（maintain_session）且之前已有执行历史，则将历史代码追加到 prompt 中
        # 形成连续上下文，使得生成代码时能参考之前的操作记录。
        if self._cfg['maintain_session'] and self.exec_hist != '': #添加历史执行记录
            prompt += f'\n{self.exec_hist}'
        
        prompt += '\n'  # separate prompted examples with the query part

        if self._cfg['include_context']:
            assert self._context is not None, 'context is None'
            prompt += f'\n{self._context}'

        user_query = f'{self._cfg["query_prefix"]}{query}{self._cfg["query_suffix"]}'
        prompt += f'\n{user_query}'

        return prompt, user_query
    
    # 对 OpenAI API 调用进行缓存。如果相同的参数已调用过，就直接返回缓存中的结果
    def _cached_api_call(self, **kwargs):
        # check whether completion endpoint or chat endpoint is used
        if kwargs['model'] != 'gpt-3.5-turbo-instruct' and \
            any([chat_model in kwargs['model'] for chat_model in ['gpt-3.5', 'gpt-4']]):
            # add special prompt for chat endpoint
            user1 = kwargs.pop('prompt')
            new_query = '# Query:' + user1.split('# Query:')[-1]
            user1 = ''.join(user1.split('# Query:')[:-1]).strip()
            user1 = f"I would like you to help me write Python code to control a robot arm operating in a tabletop environment. Please complete the code every time when I give you new query. Pay attention to appeared patterns in the given context code. Be thorough and thoughtful in your code. Do not include any import statement. Do not repeat my question. Do not provide any text explanation (comment in code is okay). I will first give you the context of the code below:\n\n```\n{user1}\n```\n\nNote that x is back to front, y is left to right, and z is bottom to up."
            assistant1 = f'Got it. I will complete what you give me next.'
            user2 = new_query
            # handle given context (this was written originally for completion endpoint)
            if user1.split('\n')[-4].startswith('objects = ['):
                obj_context = user1.split('\n')[-4]
                # remove obj_context from user1
                user1 = '\n'.join(user1.split('\n')[:-4]) + '\n' + '\n'.join(user1.split('\n')[-3:])
                # add obj_context to user2
                user2 = obj_context.strip() + '\n' + user2
            messages=[
                {"role": "system", "content": "You are a helpful assistant that pays attention to the user's instructions and writes good python code for operating a robot arm in a tabletop environment."},
                {"role": "user", "content": user1},
                {"role": "assistant", "content": assistant1},
                {"role": "user", "content": user2},
            ]
            kwargs['messages'] = messages
            if kwargs in self._cache:
                print('(using cache)', end=' ')
                return self._cache[kwargs]
            else:
                ret = openai.ChatCompletion.create(**kwargs)['choices'][0]['message']['content']
                # post processing
                ret = ret.replace('```', '').replace('python', '').strip()
                self._cache[kwargs] = ret
                return ret
        else:
            if kwargs in self._cache:
                print('(using cache)', end=' ')
                return self._cache[kwargs]
            else:
                ret = openai.Completion.create(**kwargs)['choices'][0]['text'].strip()
                self._cache[kwargs] = ret
                return ret


    # 主干执行方法
    def __call__(self, query, **kwargs):

        #调用 build_prompt 构造最终输入给 LLM 的 prompt
        prompt, user_query = self.build_prompt(query)

        #通过 _cached_api_call 调用 OpenAI API 生成代码
        start_time = time.time()
        while True:
            try:
                code_str = self._cached_api_call(
                    prompt=prompt,
                    stop=self._stop_tokens,
                    temperature=self._cfg['temperature'],
                    model=self._cfg['model'],
                    max_tokens=self._cfg['max_tokens']
                )
                break
            except (RateLimitError, APIConnectionError) as e:
                print(f'OpenAI API got err {e}')
                print('Retrying after 3s.')
                sleep(3)
        print(f'*** OpenAI API call took {time.time() - start_time:.2f}s ***')

        #根据配置决定是否将上下文 _context 与生成代码拼接，然后形成最终待执行代码字符串 to_exec 和日志记录字符串 to_log。
        #使用代码高亮（highlight 函数、PythonLexer）将日志漂亮地打印出来，显示生成代码和上下文信息
        if self._cfg['include_context']:
            assert self._context is not None, 'context is None'

            # 包含上下文：将上下文字符串 self._context 与生成的代码 code_str 拼接，获得to_exec
            to_exec = f'{self._context}\n{code_str}'

            to_log = f'{self._context}\n{user_query}\n{code_str}'
        else:

            # 不包含上下文时：直接使用生成的代码
            to_exec = code_str
            to_log = f'{user_query}\n{to_exec}'
        to_log_pretty = highlight(to_log, PythonLexer(), TerminalFormatter())
        if self._cfg['include_context']:
            print('#'*40 + f'\n## "{self._name}" generated code\n' + f'## context: "{self._context}"\n' + '#'*40 + f'\n{to_log_pretty}\n')
        else:
            print('#'*40 + f'\n## "{self._name}" generated code\n' + '#'*40 + f'\n{to_log_pretty}\n')

        gvars = merge_dicts([self._fixed_vars, self._variable_vars])
        lvars = kwargs
        '''
        收集全局变量 gvars：
    self._fixed_vars：固定变量（numpy、quaternion 计算等）
    self._variable_vars：环境相关变量（任务状态、API
收集局部变量 lvars
    
        '''
        # 对于非高层模块（即当前模块的名称不在 ['composer', 'planner'] 中），代码会进一步处理 to_exec
        # 封装为函数：在代码前添加函数定义 def ret_val():\n，使得生成的代码整体成为一个名为 ret_val 的函数体
        # return function instead of executing it so we can replan using latest obs（do not do this for high-level UIs)
        if not self._name in ['composer', 'planner']:
            to_exec = 'def ret_val():\n' + to_exec.replace('ret_val = ', 'return ')
            to_exec = to_exec.replace('\n', '\n    ')

        # 在非调试模式下，直接调用 exec_safe（一个安全执行函数）来执行生成的代码。
        if self._debug:
            # only "execute" function performs actions in environment, so we comment it out
            action_str = ['execute(']
            try:
                for s in action_str:
                    exec_safe(to_exec.replace(s, f'# {s}'), gvars, lvars)
            except Exception as e:
                print(f'Error: {e}')
                import pdb ; pdb.set_trace()
        else:
            exec_safe(to_exec, gvars, lvars)

        self.exec_hist += f'\n{to_log.strip()}'

        if self._cfg['maintain_session']:
            self._variable_vars.update(lvars)#如果 maintain_session=True，则 self._variable_vars 里存储的变量（如环境状态、执行历史）会被更新。这样，在下一次 LMP 执行时，可以保留上次执行后的变量，维持连续的推理过程。

        if self._cfg['has_return']:
            if self._name == 'parse_query_obj':
                try:
                    # there may be multiple objects returned, but we also want them to be unevaluated functions so that we can access latest obs
                    return IterableDynamicObservation(lvars[self._cfg['return_val_name']])
                except AssertionError:
                    return DynamicObservation(lvars[self._cfg['return_val_name']])
            return lvars[self._cfg['return_val_name']]


# 该函数接收一个包含多个字典的列表，并将它们合并成一个字典
# 如果不同字典中存在相同的键，后面的字典中的值会覆盖前面的值。
def merge_dicts(dicts):
    return {
        k : v 
        for d in dicts
        for k, v in d.items()
    }
    
# “安全”地执行给定的代码字符串 code_str
# 执行代码前，它会进行检查，防止代码中包含不安全的内容，并将部分危险函数（如 exec、eval）替换为空函数
def exec_safe(code_str, gvars=None, lvars=None):

    #定义了一个禁止短语列表 banned_phrases，其中包括 'import' 和 '__'
    banned_phrases = ['import', '__']
    for phrase in banned_phrases:
        assert phrase not in code_str
  
    if gvars is None:
        gvars = {}
    if lvars is None:
        lvars = {}

    # 空函数 empty_fn，该函数接收任意参数但什么也不做
    empty_fn = lambda *args, **kwargs: None

    # 使用前面定义的 merge_dicts 函数，将传入的 gvars 与一个包含禁用 exec 与 eval 的字典合并
    custom_gvars = merge_dicts([
        gvars,
        {'exec': empty_fn, 'eval': empty_fn}
    ])

    # 使用 exec 函数执行传入的代码字符串 code_str
    try:
        exec(code_str, custom_gvars, lvars)
    except Exception as e:
        print(f'Error executing code:\n{code_str}')
        raise e