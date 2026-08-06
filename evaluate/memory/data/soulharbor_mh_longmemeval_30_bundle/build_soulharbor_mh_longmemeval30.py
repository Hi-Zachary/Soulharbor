from __future__ import annotations

import hashlib
import importlib.util
import json
import random
import re
import sys
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

SEED = 20260806
ROOT = Path('/mnt/data')
OUT_JSON = ROOT / 'soulharbor_mh_longmemeval_30.json'
OUT_JSONL = ROOT / 'soulharbor_mh_longmemeval_30.jsonl'
OUT_ORACLE = ROOT / 'soulharbor_mh_longmemeval_30_oracle.json'
OUT_PROFILE = ROOT / 'soulharbor_mh_longmemeval_30_profile_gold.jsonl'
OUT_REPORT = ROOT / 'soulharbor_mh_longmemeval_30_report.json'
OUT_README = ROOT / 'soulharbor_mh_longmemeval_30_README.md'
OUT_GUIDE = ROOT / 'soulharbor_mh_longmemeval_30_EVALUATION_GUIDE.md'


def load_specs():
    source = ROOT / 'build_mental_health_long50.py'
    spec = importlib.util.spec_from_file_location('source_specs', source)
    module = importlib.util.module_from_spec(spec)
    sys.modules['source_specs'] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.SPECS


SPECS = load_specs()
# Three varied scenarios from each of the ten source categories.
SELECTED_INDICES = [
    0, 2, 4,
    5, 7, 9,
    10, 12, 14,
    15, 17, 19,
    20, 22, 24,
    25, 27, 29,
    30, 32, 34,
    35, 37, 39,
    40, 42, 44,
    45, 47, 49,
]
SELECTED = [SPECS[i] for i in SELECTED_INDICES]

QUESTION_TYPES = (
    ['single-session-user'] * 5
    + ['single-session-assistant'] * 4
    + ['single-session-preference'] * 4
    + ['multi-session'] * 5
    + ['temporal-reasoning'] * 4
    + ['knowledge-update'] * 5
    + ['abstention'] * 3
)
_rng_types = random.Random(SEED + 91)
_rng_types.shuffle(QUESTION_TYPES)

WEEKDAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

ORDINARY_EVENTS = [
    '下班时公交临时改道，我多走了两站，回家比平时晚了二十分钟。',
    '周末把床单洗了，又清理了冰箱，发现一盒酸奶已经过期。',
    '中午和同事去楼下吃面，店里换了菜单，我点了以前没吃过的番茄口味。',
    '手机系统更新后闹钟铃声变了，第二天早上我差点没反应过来。',
    '邻居白天在装修，我戴着耳机把手头不太费脑子的事情做完了。',
    '朋友发来一张新开的咖啡店照片，我们聊了几句，但没有约具体时间。',
    '快递放错了楼栋，我下楼找了一圈，后来是门卫帮我联系到的。',
    '晚上试着做了一道新菜，盐放得有点多，最后配了米饭还是吃完了。',
    '图书馆常坐的位置被占了，我换到靠楼梯的桌子，专注得比想象中好。',
    '周日去理发，店里人不多，比预计早半小时结束。',
    '我把手机相册里重复的照片删了些，留下了几组家人和旅行的。',
    '楼下便利店做活动，我买了两瓶平时常喝的无糖茶。',
    '同事过生日带了蛋糕，我吃了一小块，下午开会时大家都在聊口味。',
    '雨下得很突然，我在地铁口等了十来分钟，顺手给家里打了个电话。',
    '书桌抽屉卡住了，我花了半小时把里面的旧票据重新整理了一遍。',
    '周末和朋友看了场电影，散场后主要聊剧情，没怎么谈工作。',
    '小区停水两个小时，我提前接了两壶水，晚上做饭没有受太大影响。',
    '银行卡到期换了新卡，我把几个自动扣款重新绑定了一遍。',
    '我在超市排队时碰到以前的同学，只简单聊了近况，没有约下次见面。',
    '家里的绿植长了新叶，我换了个靠窗的位置，早上光线更好。',
    '共享单车的车锁一直打不开，我最后走了十五分钟去地铁站。',
    '午休时看完了一篇很短的访谈，里面谈的是城市里的公共长椅。',
    '外卖送错了一份配菜，商家后来退了几块钱，我没有再重新下单。',
    '我把衣柜里很久没穿的两件外套装好，准备周末送去回收点。',
    '同事临时换了会议室，我到原来的楼层才看到群里的更新。',
    '早上出门忘带门禁卡，在楼下等同事帮我开门，耽误了几分钟。',
    '家里灯泡坏了一个，我下班后买了新的，换的时候才发现型号不一样。',
    '周末去了附近的菜市场，摊主多送了两根香菜。',
    '我把常用软件的通知重新关了一遍，只留下电话和日历提醒。',
    '朋友临时取消了晚饭，我回家煮了面，顺便把第二天的早餐准备好。',
    '小区门口修路，最近去地铁站需要从另一条巷子绕过去。',
    '午后阳光很好，我出去走了一小圈，回来后继续处理原来的事情。',
    '打印机突然卡纸，我拆开后发现里面夹着半张旧标签。',
    '我在旧背包里找到一副很久没用的耳机，充电后居然还能正常工作。',
    '周六去取眼镜，店员又帮我把镜腿调紧了一点。',
    '天气降温，我把厚被子拿出来晒了晒，房间里有一股太阳味。',
    '网上买的书晚到了三天，我先借了电子版看前两章。',
    '朋友在群里分享了旅行照片，我看完后问了她那边的天气。',
    '下午有一场临时消防演练，大家在楼下站了十分钟才回去。',
    '我清理邮箱时翻到一封两年前的活动通知，顺手把几个订阅退掉了。',
]


ORDINARY_TAILS = [
    '回到家时已经快八点了。',
    '我后来给同事回了条消息。',
    '中间还接了一个家里的电话。',
    '晚饭最后是在家里简单解决的。',
    '第二天早上想起来时已经不太在意了。',
    '我顺手把第二天要带的东西也放在门口了。',
    '处理完以后，我去楼下走了十分钟。',
    '那天晚上我比平时早洗漱了一会儿。',
    '我回去后先烧了壶水。',
    '后来我把这件事讲给朋友听了一遍。',
    '我当时正准备出门，所以没有停留太久。',
    '结束后我把剩下的时间留给了做饭。',
    '我还顺便把桌上的几张纸收进了文件夹。',
    '那天风很大，回去后外套上都是灰。',
    '我没有再重新安排整晚，只把最急的一件事做完。',
    '第二天的日程没有因此取消。',
]

FILLER_ASSISTANT = [
    '这种小插曲会占掉一点精力，后来你还是把当天剩下的事情接上了吗？',
    '听起来虽然麻烦，但你处理得比较具体，没有把整天都交给这件事。',
    '普通生活里的这些细节也值得留意，它们能让一段时间不只剩下压力本身。',
    '你当时做了一个够用的处理，没有为了把事情弄得完美再增加很多步骤。',
    '这件事没有特别大的结论，不过能看出你那天还有余力照顾日常安排。',
    '有时候这种意外最让人烦的是它打断节奏，不一定是事情本身有多严重。',
    '你说得很具体。除了耽误一点时间，它后来还有影响到晚上的安排吗？',
    '至少这件事后来有一个明确的收尾，没有一直挂在你心里。',
    '听上去是很日常的一段。你愿意的话，我们继续回到最近真正让你费力的部分。',
    '这种时候能采用一个简单办法把事情结束掉，往往比反复补救更省力。',
]


FILLER_REPLY_TAILS = [
    '你后来还按原来的节奏吃饭了吗？',
    '听起来它主要占掉的是一点时间。',
    '这种收尾方式已经够用了。',
    '你没有为了补偿它再塞进更多任务。',
    '至少当天还有一部分节奏保留下来了。',
    '后来如果没有继续影响，也可以先放下。',
    '你当时的处理比较务实。',
    '这件事最后没有一直悬着。',
    '它更像是日常里的一次打断。',
    '你后来能回到手头的事情就好。',
    '先不用从这件事里得出太大的结论。',
    '我们可以把注意力再放回最近反复出现的困难。',
]

TRIGGER_USER = [
    '这周又碰到{trigger}。我最先做的是把原来的安排全部停下来，想先把不确定的地方都弄清楚，结果一晚上过去反而更乱。',
    '前几天{trigger}，我当时还没来得及判断事情有多大，就已经开始反复检查和重排计划。',
    '最近一次{trigger}时，我表面上还在做事，脑子里却一直预演最坏的情况，后来连原本简单的任务也拖住了。',
    '我发现{trigger}很容易把我带回原来的循环：先想一次性解决，再因为做不到觉得自己更差。',
]

ASSISTANT_TRIGGER = [
    '先别急着评价自己。把那次过程拆成“发生了什么、你马上做了什么、后来付出了什么代价”，会更容易看清可以调整的位置。',
    '你描述的触发点已经很明确。我们可以先看最早出现的那个动作，而不是要求自己立刻没有压力。',
    '这听起来像是节奏被突然打断后，你会用更多控制来换安全感。可以先找一个最小的环节试着松动。',
    '先把这次当作一段过程记录下来。最有用的信息通常不是“我又失败了”，而是从哪一步开始事情变得更难。',
]

ASSISTANT_GENERAL = [
    '这个安排足够具体，之后更容易判断它到底有没有帮助。',
    '先把范围定小一点，通常比同时改很多事情更容易看出效果。',
    '你可以把成功标准放在“做过一次尝试”，而不是每次都立刻感觉轻松。',
    '这个顺序把恢复空间也算进去了，不需要靠一直绷着来维持。',
    '先按这个版本走一段，再根据实际情况改，不用现在把所有以后都决定完。',
]

THIRD_PARTY_ACTIONS = [
    '把社交全部停掉一阵子',
    '每天工作到凌晨才肯休息',
    '遇到压力就连续打游戏到很晚',
    '把每件事都写进非常细的时间表',
    '什么都不说，等事情自己过去',
    '同时找很多人确认同一个决定',
]
THIRD_PARTY_NAMES = ['阿凯', '小北', '叶子', '文浩', '晨晨', '小岚', '可欣', '阿卓']


def fmt_date(dt: datetime) -> str:
    return dt.strftime('%Y/%m/%d') + f' ({WEEKDAYS[dt.weekday()]}) ' + dt.strftime('%H:%M')


def clean_sentence(text: str) -> str:
    value = re.sub(r'\s+', ' ', text).strip()
    if value and value[-1] not in '。！？!?；;：:':
        value += '。'
    return value


def make_turn(history_id: str, session_no: int, turn_no: int, role: str, content: str) -> dict[str, Any]:
    suffix = 'u' if role == 'user' else 'a'
    return {
        'message_id': f'{history_id}-s{session_no:02d}-{suffix}{turn_no}',
        'role': role,
        'content': clean_sentence(content),
    }


def make_session(history_id: str, session_no: int, dt: datetime, exchanges: list[tuple[str, str]]) -> dict[str, Any]:
    counters = {'user': 0, 'assistant': 0}
    turns = []
    for role, content in exchanges:
        counters[role] += 1
        turns.append(make_turn(history_id, session_no, counters[role], role, content))
    return {
        'session_id': f'{history_id}-s{session_no:02d}',
        'date': fmt_date(dt),
        'turns': turns,
    }


def ordinary_session(history_id: str, session_no: int, dt: datetime, rng: random.Random, used: set[str]) -> dict[str, Any]:
    choices = [x for x in ORDINARY_EVENTS if x not in used]
    if not choices:
        used.clear()
        choices = ORDINARY_EVENTS[:]
    event = rng.choice(choices)
    used.add(event)
    event = event + rng.choice(ORDINARY_TAILS)
    reply = rng.choice(FILLER_ASSISTANT) + rng.choice(FILLER_REPLY_TAILS)
    if rng.random() < 0.22:
        follow = rng.choice([
            '后来没有再出别的问题，我只是比原计划晚了一点。',
            '我回去后还是照常吃了饭，剩下的事放到第二天处理。',
            '当晚我没再继续折腾，收拾好就休息了。',
            '第二天想起来时已经没那么在意了。',
            '我把需要补的一小步做完，其他没有继续扩大。',
        ])
        return make_session(history_id, session_no, dt, [
            ('user', event), ('assistant', reply), ('user', follow),
            ('assistant', rng.choice(ASSISTANT_GENERAL)),
        ])
    return make_session(history_id, session_no, dt, [('user', event), ('assistant', reply)])


def build_sample(spec, index: int, effective_type: str) -> tuple[dict[str, Any], dict[str, Any]]:
    rng = random.Random(SEED + index * 1009)
    history_id = f'sh_mh_lme_{index:03d}'
    qid = history_id + ('_abs' if effective_type == 'abstention' else '')

    start_year = rng.choice([2023, 2024, 2025])
    start_month = rng.randint(1, 5)
    current_dt = datetime(start_year, start_month, rng.randint(2, 24), rng.choice([9, 11, 14, 19]), rng.choice([0, 15, 30, 45]))
    session_target = rng.randint(34, 38)
    used_filler: set[str] = set()

    support_text = f'{spec.support_relation}{spec.support_name}，{spec.checkin}'
    third_name = rng.choice(THIRD_PARTY_NAMES)
    third_action = rng.choice(THIRD_PARTY_ACTIONS)
    trigger_core = spec.trigger[:-1] if spec.trigger.endswith('时') else spec.trigger
    if trigger_core.startswith('一次'):
        trigger_core = trigger_core[2:]

    core: list[tuple[str, list[tuple[str, str]]]] = [
        ('background', [
            ('user', f'我最近的情况是{spec.role}。平时大部分事情还能照常做。最近最费力的是：{spec.concern}。一遇到{trigger_core}，节奏就很容易乱。'),
            ('assistant', '你已经把平时状态和容易失控的情境分开说清楚了。先从最近一次具体发生的过程开始，会比笼统判断自己更有帮助。'),
        ]),
        ('trigger', [
            ('user', rng.choice(TRIGGER_USER).format(trigger=trigger_core)),
            ('assistant', rng.choice(ASSISTANT_TRIGGER)),
        ]),
        ('support', [
            ('user', f'最近{spec.support_relation}{spec.support_name}和我约好{spec.checkin}。有时候我们只聊吃饭和最近看的东西，不一定每次都谈这件事。'),
            ('assistant', '这种固定联系的好处是不用等到特别难受才临时找人，也不会把每次联系都变成一次正式复盘。'),
        ]),
        ('old_plan', [
            ('user', f'前阵子我给自己定了个办法：{spec.old_plan}。刚开始会觉得至少抓住了点什么，但几天后反而更累。'),
            ('assistant', '这个办法短期能减少不确定感，但代价似乎是恢复时间越来越少。你可以先留意它最容易在哪个时段失控。'),
        ]),
        ('assistant_plan', [
            ('user', f'我想试一个小一点的调整，主要针对“{spec.concern}”，但不想再做一套很复杂的计划。'),
            ('assistant', spec.assistant_plan),
            ('user', '这个范围我能接受。我先做一周，看看在什么情况下有用。'),
            ('assistant', rng.choice(ASSISTANT_GENERAL)),
        ]),
        ('event_a', [
            ('user', f'{spec.event_a}是在{rng.choice(["周二", "周四", "周六"])}发生的。我把日期记在了日历里，之后才开始处理下一步。'),
            ('assistant', '有明确的时间点很好，之后回看变化时不用只凭模糊印象。'),
        ]),
        ('preference', [
            ('user', f'我慢慢发现，压力上来时更适合我的顺序是{spec.preference}。一开始就逼自己解释清楚，通常只会更乱。'),
            ('assistant', '这个顺序既具体，也给了你一点缓冲。重要的是它在真实场景里能不能重复出现。'),
        ]),
        ('boundary', [
            ('user', f'我把一个边界说得更清楚了：{spec.boundary}。这样临到当下时不用重新和自己争一遍。'),
            ('assistant', '边界一旦能说清时间和行为，通常比“我以后少做一点”更容易执行。'),
        ]),
        ('detail', [
            ('user', spec.detail_context),
            ('assistant', '这个细节很具体。把具体发生了什么留下来，比只记住“那次很糟”更有用。'),
        ]),
        ('event_b', [
            ('user', f'{spec.event_b}比前面的事情晚一些。我当时还特意翻了日历，确认两件事不是同一周发生的。'),
            ('assistant', '这样时间关系就比较清楚了，后面不容易把不同阶段的决定混在一起。'),
        ]),
        ('update', [
            ('user', f'我试了一阵后发现，{spec.old_plan}让我越来越难恢复。这周我改成{spec.new_plan}，先按这个版本执行。'),
            ('assistant', '新做法的范围更清楚，也保留了继续推进事情的空间。先观察它在忙的时候是否仍然可行。'),
        ]),
        ('support_followup', [
            ('user', f'我和{spec.support_name}的联系还在按原来的节奏继续，还是{spec.checkin}。最近一次我们聊了很多普通生活，没有专门复盘压力。'),
            ('assistant', '固定支持关系不必每次都围绕困难展开，普通交流本身也能维持连接感。'),
        ]),
        ('goal', [
            ('user', f'接下来一段时间，我想持续推进的是{spec.goal}。我希望它能成为几个月的方向，而不是只坚持一周。'),
            ('assistant', '这个目标既包含现实任务，也包含你希望保留的生活节奏，可以用来判断后面的调整是否值得。'),
        ]),
        ('current_confirmation', [
            ('user', f'昨晚忙起来时，我确实又想到过“{spec.old_plan}”，最后还是照着“{spec.new_plan}”做了。今天回头看，至少没有把原来的节奏全部推翻。'),
            ('assistant', '出现旧念头和重新采用旧办法不是一回事。你最后实际执行的行为更能说明现在的安排。'),
        ]),
        ('third_party', [
            ('user', f'朋友{third_name}最近说他遇到压力时会{third_action}。听完我能理解他的处境，但那不是我现在采用的办法。'),
            ('assistant', '别人的做法可以提供参照，但不需要自动变成你的方案。'),
        ]),
        ('setback', [
            ('user', f'这周有一天，{trigger_core}又和另外两件事挤在一起，我还是乱了一阵。晚上缓过来后，我没有临时增加新规则。'),
            ('assistant', '一次波动不必立刻变成新的结论。更值得看的是你后来用了什么方式回到原来的安排。'),
            ('user', f'我最后还是{spec.preference}，然后只处理了最紧急的一项。'),
            ('assistant', '这说明那个顺序已经不只是纸面上的想法，至少在一次真实场景里派上了用场。'),
        ]),
    ]

    # Six stages preserve narrative dependencies while varying local order.
    stages = [
        ['background', 'trigger', 'support'],
        ['old_plan', 'assistant_plan'],
        ['event_a', 'preference', 'third_party'],
        ['boundary', 'detail', 'setback'],
        ['event_b', 'update', 'support_followup'],
        ['goal', 'current_confirmation'],
    ]
    core_map = {k: v for k, v in core}

    sessions: list[dict[str, Any]] = []
    key_to_session: dict[str, dict[str, Any]] = {}
    key_to_turns: dict[str, list[dict[str, Any]]] = {}

    def advance() -> datetime:
        nonlocal current_dt
        current_dt += timedelta(days=rng.randint(4, 11), hours=rng.choice([-1, 0, 0, 1]))
        return current_dt

    # Put 2-4 natural sessions between most evidence-bearing sessions.
    for stage_index, stage_keys in enumerate(stages):
        if stage_index > 0:
            for _ in range(rng.randint(1, 2)):
                no = len(sessions) + 1
                sessions.append(ordinary_session(history_id, no, advance(), rng, used_filler))
        local_keys = stage_keys[:]
        # Preserve critical ordering inside stage 5: event_b before update.
        if stage_index not in {0, 4, 5}:
            rng.shuffle(local_keys)
        for key in local_keys:
            no = len(sessions) + 1
            sess = make_session(history_id, no, advance(), core_map[key])
            sessions.append(sess)
            key_to_session[key] = sess
            key_to_turns[key] = sess['turns']
            if rng.random() < 0.35:
                no = len(sessions) + 1
                sessions.append(ordinary_session(history_id, no, advance(), rng, used_filler))

    while len(sessions) < session_target:
        no = len(sessions) + 1
        sessions.append(ordinary_session(history_id, no, advance(), rng, used_filler))

    # If stochastic interleaving exceeded the target slightly, keep it; max is checked below.
    question_dt = current_dt + timedelta(days=rng.randint(6, 12), hours=1)

    def first_turn(key: str, role: str = 'user') -> dict[str, Any]:
        for t in key_to_turns[key]:
            if t['role'] == role:
                return t
        raise KeyError((key, role))

    evidence_turns: list[dict[str, Any]] = []
    answer_sessions: list[str] = []
    superseded: list[str] = []
    memory_target = 'episodic'
    aliases: list[str] = []
    answer_fields: dict[str, str] | None = None
    evaluation: dict[str, Any]
    answerable = True

    if effective_type == 'single-session-user':
        question = spec.detail_question
        answer = spec.detail_answer
        aliases = [spec.detail_answer]
        evidence_turns = [first_turn('detail')]
        answer_sessions = [key_to_session['detail']['session_id']]
        evaluation = {'type': 'semantic_short_answer'}
    elif effective_type == 'single-session-assistant':
        question = [
            '助手当时建议用户先做哪项小范围尝试？',
            '为了不把调整做得太复杂，助手给出的具体试行方法是什么？',
            '助手在那次讨论中提出了什么可执行的尝试？',
            '用户请求一个小调整时，助手具体建议了什么？',
        ][(index - 1) % 4]
        answer = spec.assistant_plan
        aliases = [spec.assistant_plan]
        evidence_turns = [first_turn('assistant_plan', 'assistant')]
        answer_sessions = [key_to_session['assistant_plan']['session_id']]
        evaluation = {'type': 'semantic_short_answer'}
    elif effective_type == 'single-session-preference':
        question = [
            '用户明确表示，压力上来时更适合自己的处理顺序是什么？',
            '面对压力时，用户认为怎样的先后顺序更适合自己？',
            '用户提到自己在压力升高时更愿意先怎么处理？',
            '用户后来确认，哪种应对顺序对自己更合适？',
        ][(index - 1) % 4]
        answer = spec.preference
        aliases = [spec.preference]
        evidence_turns = [first_turn('preference')]
        answer_sessions = [key_to_session['preference']['session_id']]
        memory_target = 'profile'
        evaluation = {'type': 'semantic_short_answer'}
    elif effective_type == 'multi-session':
        detail_prompt = spec.detail_question.rstrip('？?')
        question = [
            f'请分别回答三个问题：用户和谁保持怎样的固定联系？{spec.detail_question}助手建议先试行的具体方法是什么？',
            f'综合不同阶段的对话，请说明用户的固定支持安排；回答“{spec.detail_question}”；并写出助手当时给出的试行方法。',
            f'请从三段不同的对话中找出：固定联系对象与频率、{detail_prompt}的答案，以及助手建议的小范围尝试。',
            f'用户的固定支持安排是什么？另外，{spec.detail_question}最后，助手提出了什么具体尝试？',
            f'请依次给出固定支持安排、相关事件的具体细节（{detail_prompt}）和助手的试行建议。',
        ][(index - 1) % 5]
        answer_fields = {
            'support_arrangement': support_text,
            'event_detail': spec.detail_answer,
            'assistant_plan': spec.assistant_plan,
        }
        answer = f'固定联系是{support_text}；相关具体细节是{spec.detail_answer}；助手建议{spec.assistant_plan}。'
        evidence_turns = [first_turn('support'), first_turn('detail'), first_turn('assistant_plan', 'assistant')]
        answer_sessions = [key_to_session[k]['session_id'] for k in ['support', 'detail', 'assistant_plan']]
        memory_target = 'both'
        evaluation = {'type': 'structured_fields', 'required_fields': list(answer_fields)}
    elif effective_type == 'temporal-reasoning':
        question = f'“{spec.event_a}”和“{spec.event_b}”哪一件先发生？请按先后顺序回答。'
        answer = f'先发生“{spec.event_a}”，之后发生“{spec.event_b}”。'
        aliases = [f'先{spec.event_a}，后{spec.event_b}', f'{spec.event_a}在前，{spec.event_b}在后']
        evidence_turns = [first_turn('event_a'), first_turn('event_b')]
        answer_sessions = [key_to_session['event_a']['session_id'], key_to_session['event_b']['session_id']]
        evaluation = {'type': 'temporal_order', 'first': spec.event_a, 'second': spec.event_b}
    elif effective_type == 'knowledge-update':
        question = [
            '用户目前实际采用的主要安排是什么？',
            '经过后续调整，用户现在执行的是哪一套安排？',
            '用户早期的做法后来发生了变化；当前有效的安排是什么？',
            '根据较新的对话，用户现在把什么作为主要方案？',
            '用户目前没有继续沿用最初的做法。现在实际采用什么安排？',
        ][(index - 1) % 5]
        answer = spec.new_plan
        aliases = [spec.new_plan]
        evidence_turns = [first_turn('update'), first_turn('current_confirmation')]
        answer_sessions = [key_to_session['update']['session_id'], key_to_session['current_confirmation']['session_id']]
        superseded = [first_turn('old_plan')['message_id']]
        memory_target = 'both'
        evaluation = {'type': 'knowledge_update'}
    elif effective_type == 'abstention':
        question = spec.abstention_question
        answer = '历史中没有提供相关信息。'
        aliases = ['历史未提供', '对话中没有说明', '无法从现有历史判断', '没有足够信息']
        answerable = False
        evaluation = {'type': 'abstention'}
    else:
        raise ValueError(effective_type)

    evidence_ids = [t['message_id'] for t in evidence_turns]
    evidence_set = set(evidence_ids)
    for session in sessions:
        for turn in session['turns']:
            if turn['message_id'] in evidence_set:
                turn['has_answer'] = True

    sample = {
        'question_id': qid,
        'question_type': 'single-session-user' if effective_type == 'abstention' else effective_type,
        'question': question,
        'answer': answer,
        'question_date': fmt_date(question_dt),
        'haystack_session_ids': [s['session_id'] for s in sessions],
        'haystack_dates': [s['date'] for s in sessions],
        'haystack_sessions': [s['turns'] for s in sessions],
        'answer_session_ids': answer_sessions,
        # SoulHarbor extensions below.
        'history_id': history_id,
        'capability': effective_type,
        'domain': 'mental-health-adjacent-longitudinal-support',
        'category': spec.category,
        'theme': spec.theme,
        'memory_target': memory_target,
        'answerable': answerable,
        'aliases': aliases,
        'evidence_message_ids': evidence_ids,
        'superseded_message_ids': superseded,
        'evaluation': evaluation,
        'metadata': {
            'language': 'zh-CN',
            'synthetic': True,
            'contains_diagnosis': False,
            'contains_self_harm': False,
            'session_count': len(sessions),
        },
    }
    if answer_fields is not None:
        sample['answer_fields'] = answer_fields

    # Gold profile labels use only facts actually stated by the user.
    profile = {
        'history_id': history_id,
        'question_id': qid,
        'active_facts': [
            {
                'fact_id': f'{history_id}-p1',
                'content': f'用户目前采用{spec.new_plan}。',
                'evidence_message_ids': [first_turn('update')['message_id'], first_turn('current_confirmation')['message_id']],
            },
            {
                'fact_id': f'{history_id}-p2',
                'content': f'用户压力上来时更适合{spec.preference}。',
                'evidence_message_ids': [first_turn('preference')['message_id']],
            },
            {
                'fact_id': f'{history_id}-p3',
                'content': f'用户与{spec.support_relation}{spec.support_name}保持固定联系：{spec.checkin}。',
                'evidence_message_ids': [first_turn('support')['message_id'], first_turn('support_followup')['message_id']],
            },
            {
                'fact_id': f'{history_id}-p4',
                'content': f'用户的边界是{spec.boundary}。',
                'evidence_message_ids': [first_turn('boundary')['message_id']],
            },
            {
                'fact_id': f'{history_id}-p5',
                'content': f'用户持续推进的目标是{spec.goal}。',
                'evidence_message_ids': [first_turn('goal')['message_id']],
            },
        ],
        'superseded_facts': [
            {
                'content': f'用户采用{spec.old_plan}。',
                'evidence_message_ids': [first_turn('old_plan')['message_id']],
            }
        ],
        'third_party_facts': [
            {
                'content': f'{third_name}遇到压力时会{third_action}。',
                'evidence_message_ids': [first_turn('third_party')['message_id']],
            }
        ],
        'transient_message_ids': [
            s['turns'][0]['message_id']
            for s in sessions
            if s['session_id'] not in set(key_to_session[k]['session_id'] for k in key_to_session)
        ][:3],
    }
    return sample, profile


def make_oracle(sample: dict[str, Any]) -> dict[str, Any]:
    keep = set(sample['answer_session_ids'])
    sessions = []
    ids = []
    dates = []
    for sid, date, turns in zip(sample['haystack_session_ids'], sample['haystack_dates'], sample['haystack_sessions']):
        if sid in keep:
            ids.append(sid)
            dates.append(date)
            sessions.append(turns)
    oracle = dict(sample)
    oracle['haystack_session_ids'] = ids
    oracle['haystack_dates'] = dates
    oracle['haystack_sessions'] = sessions
    oracle['metadata'] = dict(sample['metadata'], oracle=True)
    return oracle


def validate(samples: list[dict[str, Any]], profiles: list[dict[str, Any]]) -> dict[str, Any]:
    assert len(samples) == 30
    assert len(profiles) == 30
    assert len({x['question_id'] for x in samples}) == 30
    type_counts = Counter(x['capability'] for x in samples)
    assert type_counts == Counter({
        'single-session-user': 5,
        'single-session-assistant': 4,
        'single-session-preference': 4,
        'multi-session': 5,
        'temporal-reasoning': 4,
        'knowledge-update': 5,
        'abstention': 3,
    }), type_counts

    session_counts = []
    message_counts = []
    evidence_counts = []
    answer_positions = []
    category_counts = Counter()
    exact_session_texts = Counter()
    forbidden_phrases = [
        '这只是一个普通生活片段',
        '不代表新的习惯',
        '没有改变我当前的主要安排',
        '不应作为当前安排',
    ]
    forbidden_hits = []

    for sample in samples:
        category_counts[sample['category']] += 1
        ids = sample['haystack_session_ids']
        dates = sample['haystack_dates']
        sessions = sample['haystack_sessions']
        assert len(ids) == len(dates) == len(sessions)
        assert 34 <= len(ids) <= 40, len(ids)
        session_counts.append(len(ids))
        assert len(ids) == len(set(ids))
        dt_values = [datetime.strptime(x, '%Y/%m/%d (%a) %H:%M') for x in dates]
        assert dt_values == sorted(dt_values)
        qdt = datetime.strptime(sample['question_date'], '%Y/%m/%d (%a) %H:%M')
        assert qdt > dt_values[-1]

        mids = []
        evidence_from_turns = []
        for turns in sessions:
            session_text = '\n'.join(f"{t['role']}: {t['content']}" for t in turns)
            exact_session_texts[session_text] += 1
            for t in turns:
                mids.append(t['message_id'])
                if t.get('has_answer'):
                    evidence_from_turns.append(t['message_id'])
                for phrase in forbidden_phrases:
                    if phrase in t['content']:
                        forbidden_hits.append((sample['question_id'], phrase, t['message_id']))
        assert len(mids) == len(set(mids))
        message_counts.append(len(mids))
        assert set(sample['evidence_message_ids']) == set(evidence_from_turns)
        assert set(sample['answer_session_ids']) <= set(ids)
        assert set(sample['superseded_message_ids']) <= set(mids)
        evidence_counts.append(len(sample['evidence_message_ids']))
        for sid in sample['answer_session_ids']:
            answer_positions.append(ids.index(sid) / max(1, len(ids) - 1))

        if sample['capability'] == 'abstention':
            assert sample['question_id'].endswith('_abs')
            assert sample['answerable'] is False
            assert sample['answer_session_ids'] == []
            assert sample['evidence_message_ids'] == []
        else:
            assert sample['answerable'] is True
            assert sample['answer_session_ids']
            assert sample['evidence_message_ids']
            assert not sample['question_id'].endswith('_abs')

        if sample['capability'] == 'single-session-assistant':
            evidence_roles = [
                t['role'] for turns in sessions for t in turns
                if t['message_id'] in set(sample['evidence_message_ids'])
            ]
            assert evidence_roles == ['assistant']
        if sample['capability'] in {'single-session-user', 'single-session-preference'}:
            evidence_roles = [
                t['role'] for turns in sessions for t in turns
                if t['message_id'] in set(sample['evidence_message_ids'])
            ]
            assert evidence_roles == ['user']
        if sample['capability'] == 'knowledge-update':
            assert sample['superseded_message_ids']
            mid_to_pos = {t['message_id']: i for i, turns in enumerate(sessions) for t in turns}
            assert max(mid_to_pos[x] for x in sample['superseded_message_ids']) < min(mid_to_pos[x] for x in sample['evidence_message_ids'])
        assert sample['answer'] not in sample['question']

    assert not forbidden_hits, forbidden_hits[:5]
    duplicate_sessions = sum(v - 1 for v in exact_session_texts.values() if v > 1)
    # Repeated assistant-only wording can still occur, but whole sessions should be nearly unique.
    assert duplicate_sessions <= 10, duplicate_sessions

    raw_jsonl = ''.join(json.dumps(x, ensure_ascii=False, separators=(',', ':')) + '\n' for x in samples)
    return {
        'schema_version': '1.0-longmemeval-compatible',
        'instance_count': len(samples),
        'question_type_counts': dict(type_counts),
        'category_counts': dict(category_counts),
        'session_count_total': sum(session_counts),
        'sessions_per_instance': {
            'min': min(session_counts),
            'max': max(session_counts),
            'avg': round(sum(session_counts) / len(session_counts), 3),
        },
        'message_count_total': sum(message_counts),
        'messages_per_instance_avg': round(sum(message_counts) / len(message_counts), 3),
        'evidence_messages_per_question_avg': round(sum(evidence_counts) / len(evidence_counts), 3),
        'mean_evidence_session_relative_position': round(sum(answer_positions) / len(answer_positions), 3),
        'all_questions_open_ended': True,
        'all_dates_sorted': True,
        'all_question_dates_after_history': True,
        'all_evidence_ids_valid': True,
        'explicit_label_language_removed': True,
        'whole_session_duplicate_count': duplicate_sessions,
        'sha256_jsonl': hashlib.sha256(raw_jsonl.encode('utf-8')).hexdigest(),
    }


def write_readme(report: dict[str, Any]) -> None:
    OUT_README.write_text(f'''# SoulHarbor-MH-LongMemEval-30

Chinese, mental-health-adjacent, question-centric long-term conversational memory benchmark.

## Files

- `soulharbor_mh_longmemeval_30.json`: main benchmark as a JSON array.
- `soulharbor_mh_longmemeval_30.jsonl`: the same 30 instances in JSONL.
- `soulharbor_mh_longmemeval_30_oracle.json`: evidence-session-only oracle histories.
- `soulharbor_mh_longmemeval_30_profile_gold.jsonl`: optional SoulHarbor profile-maintenance gold labels.
- `soulharbor_mh_longmemeval_30_report.json`: structural validation and checksums.
- `soulharbor_mh_longmemeval_30_EVALUATION_GUIDE.md`: evaluator migration notes.

## Design

- 30 independent question instances; each question owns one timestamped long history.
- {report['session_count_total']} sessions in total; {report['sessions_per_instance']['min']}-{report['sessions_per_instance']['max']} sessions per instance, average {report['sessions_per_instance']['avg']}.
- {report['message_count_total']} total user/assistant turns, average {report['messages_per_instance_avg']} turns per instance.
- Chinese natural dialogue about study, work, relationships, caregiving, adjustment, grief, performance pressure, health routines, career uncertainty and overcommitment.
- No self-harm content and no synthetic clinical diagnosis.
- No multiple-choice questions.
- No dialogue statements that explicitly label a memory as transient, stable, stale, or profile-worthy.
- Evidence sessions are not concentrated at the end of the history.

## Ability mix

```json
{json.dumps(report['question_type_counts'], ensure_ascii=False, indent=2)}
```

Abstention follows LongMemEval's naming convention: the `question_id` ends in `_abs`. The additional `capability` field explicitly records `abstention` for local aggregation.

## LongMemEval-compatible fields

- `question_id`
- `question_type`
- `question`
- `answer`
- `question_date`
- `haystack_session_ids`
- `haystack_dates`
- `haystack_sessions`
- `answer_session_ids`
- evidence turns contain `has_answer: true`

## SoulHarbor extensions

- `history_id`
- `capability`
- `memory_target`
- `answerable`
- `aliases`
- `evidence_message_ids`
- `superseded_message_ids`
- `evaluation`
- `answer_fields` for structured multi-session questions

## Important ingestion rule

`has_answer`, `message_id`, and all gold fields are evaluator metadata. Strip them before sending turns to the assistant model. Keep an internal mapping from source `message_id` to the database message ID for retrieval evaluation.

## Validation

```json
{json.dumps(report, ensure_ascii=False, indent=2)}
```
''', encoding='utf-8')


def write_guide() -> None:
    OUT_GUIDE.write_text('''# Evaluating SoulHarbor on SoulHarbor-MH-LongMemEval-30

## 1. Load one question instance at a time

Each record already contains the complete history for one question. Create a fresh database or isolated user namespace for every `question_id`.

```python
for item in dataset:
    engine = make_clean_engine(question_id=item["question_id"])
    ingest_history(engine, item)
    retrieval = engine.build_context_with_details(
        user_id=1,
        conversation_id=0,
        current_user_message=item["question"],
        recent_messages=[],
        conversation_summary=None,
        exclude_message_ids=set(),
    )
    hypothesis = reader_answer(item["question"], retrieval.context)
```

Do not share a database between question instances. This follows LongMemEval's question-centric setup and prevents facts from one synthetic user leaking into another.

## 2. Use the real timestamps

Parse `haystack_dates` and assign the same timestamp to every turn in the corresponding session. Do not convert session order into arbitrary week numbers.

```python
from datetime import datetime

def parse_lme_date(value: str) -> int:
    return int(datetime.strptime(value, "%Y/%m/%d (%a) %H:%M").timestamp())
```

## 3. Never expose annotation fields

Before ingesting a turn, send only `role` and `content` to the memory system. Do not include:

- `has_answer`
- `message_id`
- `answer_session_ids`
- `evidence_message_ids`
- `superseded_message_ids`
- `answer`, `aliases`, or `evaluation`

Keep `message_id` only in the evaluator's mapping table.

## 4. Preserve source-to-database message mappings

```python
source_to_db: dict[str, int] = {}
db_to_source: dict[int, str] = {}

for source_turn in session:
    db_id = allocate_integer_message_id()
    source_to_db[source_turn["message_id"]] = db_id
    db_to_source[db_id] = source_turn["message_id"]
    ingest(role=source_turn["role"], content=source_turn["content"], message_id=db_id)
```

The retrieval API should return selected database message IDs or anchor IDs. Convert them back to source IDs before scoring.

## 5. Output format

Keep a LongMemEval-compatible prediction file:

```json
{"question_id":"sh_mh_lme_001","hypothesis":"模型回答"}
```

For diagnostics, write a second JSONL file:

```json
{
  "question_id": "sh_mh_lme_001",
  "hypothesis": "模型回答",
  "retrieved_message_ids": ["..."],
  "retrieved_session_ids": ["..."],
  "active_profiles": ["..."],
  "retrieval_trace": {},
  "latency_ms": 0
}
```

## 6. Retrieval metrics

Calculate retrieval metrics only for answerable questions.

```python
gold_messages = set(item["evidence_message_ids"])
retrieved_messages = set(result["retrieved_message_ids"])
message_recall = len(gold_messages & retrieved_messages) / len(gold_messages)
all_evidence = gold_messages <= retrieved_messages

gold_sessions = set(item["answer_session_ids"])
retrieved_sessions = set(result["retrieved_session_ids"])
session_recall = len(gold_sessions & retrieved_sessions) / len(gold_sessions)
```

Report:

- Message Recall@K
- Session Recall@K
- Any-evidence Recall
- All-evidence Recall
- MRR for single-evidence questions

For `knowledge-update`, also report:

```python
old = set(item["superseded_message_ids"])
stale_only = bool(old & retrieved_messages) and not bool(gold_messages & retrieved_messages)
```

Retrieving both old and current evidence is not automatically an error. `stale_only` is the dangerous case.

## 7. QA scoring by evaluation type

Do not send every answer directly to one generic judge.

### semantic_short_answer

1. normalize punctuation and whitespace;
2. check `answer` and `aliases`;
3. call a semantic judge only if deterministic matching fails.

### temporal_order

Parse or judge the two ordered events using `evaluation.first` and `evaluation.second`.

### knowledge_update

The answer must state the current value. Mentioning only a superseded value is incorrect. Mentioning the old value as historical context is allowed if the current value is clear.

### structured_fields

Score every key in `answer_fields` independently and also report exact all-fields accuracy.

### abstention

Accept an explicit statement that the history does not provide enough information. Reject invented concrete answers.

## 8. Oracle run

Run the same reader on `soulharbor_mh_longmemeval_30_oracle.json`.

- Oracle wrong: likely a question/gold/reader/judge problem.
- Oracle right, full system wrong, evidence absent: retrieval problem.
- Oracle right, evidence retrieved, full system wrong: reader or context-formatting problem.

## 9. Suggested report

```json
{
  "qa": {
    "overall_accuracy": 0.0,
    "by_capability": {},
    "by_memory_target": {},
    "structured_field_accuracy": 0.0,
    "abstention_accuracy": 0.0
  },
  "retrieval": {
    "message_recall_at_k": 0.0,
    "session_recall_at_k": 0.0,
    "all_evidence_recall": 0.0,
    "stale_only_rate": 0.0
  },
  "oracle": {
    "qa_accuracy": 0.0
  },
  "profile": {
    "precision": 0.0,
    "recall": 0.0,
    "f1": 0.0,
    "stale_capture_rate": 0.0,
    "third_party_capture_rate": 0.0,
    "transient_capture_rate": 0.0
  }
}
```

## 10. Profile evaluation

Use the separate profile gold file. Compare only active atomic profiles from the profile store. Do not mix episodic chunks into the predicted profile set.

Use one-to-one semantic matching between predicted profiles and `active_facts`. Separately test whether predicted profiles match `superseded_facts`, `third_party_facts`, or the messages listed under `transient_message_ids`.
''', encoding='utf-8')


def main() -> None:
    samples = []
    profiles = []
    for i, (spec, qtype) in enumerate(zip(SELECTED, QUESTION_TYPES), start=1):
        sample, profile = build_sample(spec, i, qtype)
        samples.append(sample)
        profiles.append(profile)

    report = validate(samples, profiles)
    OUT_JSON.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding='utf-8')
    OUT_JSONL.write_text(''.join(json.dumps(x, ensure_ascii=False, separators=(',', ':')) + '\n' for x in samples), encoding='utf-8')
    OUT_ORACLE.write_text(json.dumps([make_oracle(x) for x in samples], ensure_ascii=False, indent=2), encoding='utf-8')
    OUT_PROFILE.write_text(''.join(json.dumps(x, ensure_ascii=False, separators=(',', ':')) + '\n' for x in profiles), encoding='utf-8')

    report['sha256_json'] = hashlib.sha256(OUT_JSON.read_bytes()).hexdigest()
    report['sha256_jsonl'] = hashlib.sha256(OUT_JSONL.read_bytes()).hexdigest()
    report['sha256_oracle'] = hashlib.sha256(OUT_ORACLE.read_bytes()).hexdigest()
    report['sha256_profile_gold'] = hashlib.sha256(OUT_PROFILE.read_bytes()).hexdigest()
    OUT_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    write_readme(report)
    write_guide()
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
