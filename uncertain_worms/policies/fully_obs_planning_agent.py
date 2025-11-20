import logging
import os
import random

# import traceback
from typing import Any, Dict, List, Optional, Tuple

from uncertain_worms.planners.base_planner import FullyObservablePlanner
from uncertain_worms.policies.base_policy import (
    PROMPT_DIR,
    Policy,
    requery,
    reward_model_translator,
    transition_model_translator,
)
from uncertain_worms.structs import (
    ActType,
    ObsType,
    ReplayBuffer,
    StateType,
    Transition,
)
from uncertain_worms.utils import (
    PROJECT_ROOT,
    REWARD_FUNCTION_NAME,
    TRANSITION_FUNCTION_NAME,
)

log = logging.getLogger(__name__)


def print_diff(prev_state: StateType, action: ActType, next_state: StateType) -> str:
    print_str = f"The action {str(action)} transforms the state from\n"
    print_str += str(prev_state)
    print_str += "to\n"
    print_str += str(next_state)
    return print_str


class FullyObsPlanningAgent(Policy[StateType, ActType, ObsType]):
    def __init__(
        self,
        planner: FullyObservablePlanner[StateType, ActType],
        *args: Any,
        env_description: str = "",
        iterations: int = 1000,
        num_model_attempts: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        assert self.fully_obs
        self.env_description = env_description
        self.steps_taken = 0
        self.execution_errors: Dict[str, str] = {}
        self.num_model_attempts = num_model_attempts
        self.iterations = iterations
        self.planner = planner

    def evaluate_models(self, replay_buffer: ReplayBuffer) -> Tuple[Tuple, Tuple]:
        correct_transitions = []
        incorrect_transitions = []
        correct_rewards = []
        incorrect_rewards = []

        for episode in replay_buffer.episodes:
            for previous_state, action, next_state, reward, terminated in zip(
                episode.previous_states,
                episode.actions,
                episode.next_states,
                episode.rewards,
                episode.terminated,
            ):
                try:
                    pred_reward, pred_terminated = self.planner.reward_model(
                        previous_state, action, next_state
                    )
                    rexperience = (
                        (previous_state, action, next_state),
                        (reward, terminated),
                        (pred_reward, pred_terminated),
                    )
                    if pred_reward != reward or pred_terminated != terminated:
                        incorrect_rewards.append(rexperience)
                    else:
                        correct_rewards.append(rexperience)

                except Exception as e:
                    log.info("Bug during reward evaluation")
                    # log.info(str(traceback.format_exc()))
                    log.info(f"{type(e).__name__}: {str(e)}")
                    self.execution_errors["reward_model"] = str(e)

            # Eval transition model
            for previous_state, action, next_state in zip(
                episode.previous_states, episode.actions, episode.next_states
            ):
                try:
                    pred_next_state = self.planner.transition_model(
                        previous_state, action
                    )
                    texperience = (
                        (previous_state, action),
                        (next_state,),
                        (pred_next_state,),
                    )
                    if str(pred_next_state) != str(next_state):
                        incorrect_transitions.append(texperience)
                    else:
                        correct_transitions.append(texperience)
                except Exception as e:
                    log.info("Bug during transition evaluation")
                    # log.info(str(traceback.format_exc()))
                    log.info(f"{type(e).__name__}: {str(e)}")
                    self.execution_errors["transition_model"] = str(e)

        return (
            (correct_transitions, incorrect_transitions),
            (correct_rewards, incorrect_rewards),
        )

    def update_models(self, replay_buffer: ReplayBuffer, iter_num: int) -> None:
        _ = self.evaluate_models(replay_buffer)
        self.init_types(replay_buffer)

    def reset(self) -> None:
        self.steps_taken = 0

    def get_next_action(self, state: StateType) -> ActType:
        steps_left = self.max_steps - self.steps_taken

        action, error = self.planner.plan_next_action(state, steps_left)

        self.steps_taken += 1
        if len(error) > 0:
            log.info("Planning error, taking random action")
            return random.choice(self.actions)
        else:
            return action


def get_experience_strings_for_prompts(
    transitions: List[Transition],
) -> Tuple[str, str]:
    reward_experiences_str = ""
    transition_experiences_str = ""

    for t in transitions:
        prev_state, action, state, reward, terminated = (
            t.prev_state,
            t.action,
            t.next_state,
            t.reward,
            t.terminated,
        )

        diff = print_diff(prev_state, action, state)
        reward_experiences_str += diff
        reward_experiences_str += f", the returned reward is ` {reward} ` and the returned terminated is ` {terminated} `\n\n"
        transition_experiences_str += diff
        transition_experiences_str += "\n"

    return reward_experiences_str, transition_experiences_str


class LLM_TR_FullyObsPlanningAgent(FullyObsPlanningAgent[StateType, ActType, ObsType]):
    def __init__(
        self,
        *args: Any,
        env_code_path: str = "",
        dataset_path: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.replay_buffer = ReplayBuffer[StateType, ActType, ObsType]()
        self.previous_state: Optional[StateType] = None

        self.current_reward_code: Optional[str] = EMPTY_REWARD_FUNCTION
        self.current_transition_code: Optional[str] = EMPTY_TRANSITION_FUNCTION

        self.env_code_path = env_code_path
        with open(os.path.join(PROJECT_ROOT, env_code_path, "api.py")) as f:
            self.code_api = f.read().strip()

        self.transition_messages: List[Dict[str, str]] = []
        self.reward_messages: List[Dict[str, str]] = []

        # Train
        train_replay_buffer = ReplayBuffer[StateType, int, ObsType].load_from_file(
            os.path.join(PROJECT_ROOT, dataset_path)
        )
        self.update_models(replay_buffer=train_replay_buffer, iter_num=0)

    def get_incorrect_reward_prompt(
        self,
        correct_rewards: List[Tuple],
        incorrect_rewards: List[Tuple],
    ) -> str:
        (
            (prev_state, action, next_state),
            (reward, terminated),
            (pred_reward, pred_terminated),
        ) = random.choice(incorrect_rewards)

        incorrect_exp = print_diff(prev_state, action, next_state)
        incorrect_exp += f", the returned reward should be ` {reward} ` and the returned done should be ` {terminated} `.\n"
        incorrect_exp += f"However, the implementation is wrong because it returns the predicted reward as ` {pred_reward} ` and the predicted done as ` {pred_terminated} ` instead of the correct reward as ` {reward} ` and the correct done ` {terminated} `.\n"

        (
            (prev_state, action, next_state),
            (reward, terminated),
            (pred_reward, pred_terminated),
        ) = random.choice(correct_rewards)

        correct_exp = print_diff(prev_state, action, next_state)
        correct_exp += f", the returned reward is ` {reward} ` and the returned done is ` {terminated} `.\n"

        reward_file_path = os.path.join(
            PROMPT_DIR, "refining_reward_function_prompt_hao.txt"
        )

        with open(reward_file_path, "r", encoding="utf-8") as file:
            reward_prompt = file.read()

        reward_templates = {
            "code_api": self.code_api,
            "code": self.current_reward_code,
            "correct_experience": correct_exp,
            "incorrect_experience": incorrect_exp,
            "env_description": self.env_description,
        }

        reward_prompt = reward_prompt.format(**reward_templates)

        return reward_prompt

    def get_incorrect_transition_prompt(
        self,
        correct_transitions: List[Tuple],
        incorrect_transitions: List[Tuple],
    ) -> str:
        (prev_state, action), (next_state,), (pred_next_state,) = random.choice(
            incorrect_transitions
        )
        incorrect_exp = print_diff(prev_state, action, next_state)

        incorrect_exp += (
            "However, the implementation is wrong because it returns state as\n"
        )

        incorrect_exp += str(pred_next_state)

        if len(correct_transitions) > 0:
            (prev_state, action), (next_state,), _ = random.choice(correct_transitions)
            correct_exp = print_diff(prev_state, action, next_state)
        else:
            correct_exp = ""

        transition_file_path = os.path.join(
            PROMPT_DIR, "refining_transition_function_prompt_hao.txt"
        )

        with open(transition_file_path, "r", encoding="utf-8") as file:
            transition_prompt = file.read()

        transition_templates = {
            "code_api": self.code_api,
            "code": self.current_transition_code,
            "correct_experience": correct_exp,
            "incorrect_experience": incorrect_exp,
            "env_description": self.env_description,
        }

        transition_prompt = transition_prompt.format(**transition_templates)

        return transition_prompt

    def get_starting_prompts(
        self, reward_exp: str, transition_exp: str
    ) -> Tuple[str, str]:
        reward_file_path = os.path.join(PROMPT_DIR, "reward_function_prompt.txt")

        with open(reward_file_path, "r", encoding="utf-8") as file:
            reward_prompt = file.read()

        reward_code_file_path = os.path.join(PROMPT_DIR, "reward_function_template.txt")

        with open(reward_code_file_path, "r", encoding="utf-8") as file:
            reward_code_template = file.read()

        reward_templates = {
            "exp": reward_exp,
            "code_template": reward_code_template,
            "code_api": self.code_api,
            "env_description": self.env_description,
        }
        reward_prompt = reward_prompt.format(**reward_templates)

        transition_file_path = os.path.join(
            PROMPT_DIR, "transition_function_prompt.txt"
        )

        with open(transition_file_path, "r", encoding="utf-8") as file:
            transition_prompt = file.read()

        transition_code_file_path = os.path.join(
            PROMPT_DIR, "transition_function_template.txt"
        )

        with open(transition_code_file_path, "r", encoding="utf-8") as file:
            transition_code_template = file.read()

        transition_templates = {
            "exp": transition_exp,
            "code_template": transition_code_template,
            "code_api": self.code_api,
            "env_description": self.env_description,
        }

        transition_prompt = transition_prompt.format(**transition_templates)

        return reward_prompt, transition_prompt

    def update_models(self, replay_buffer: ReplayBuffer, iter_num: int = 0) -> None:
        self.init_types(replay_buffer)

        for step in range(self.num_model_attempts):
            self.step_num = step
            to_update = self._update_models(replay_buffer, iter_num=iter_num)
            if not to_update:
                break

        self.reset_experiences()
        _ = self.evaluate_models(replay_buffer)

    def _update_models(self, replay_buffer: ReplayBuffer, iter_num: int) -> bool:
        (correct_transitions, incorrect_transitions), (
            correct_rewards,
            incorrect_rewards,
        ) = self.evaluate_models(replay_buffer)

        to_update = False
        if "transition_model" in self.execution_errors:
            log.info(
                f'Iter {iter_num}, step {self.step_num}: Transition model bug: {self.execution_errors["transition_model"]}'
            )
            to_update = True
        else:
            log.info(
                f"Iter {iter_num}, step {self.step_num}: Transition model pass rate: {len(correct_transitions) / (len(correct_transitions) + len(incorrect_transitions))} of {len(correct_transitions) + len(incorrect_transitions)}"
            )
            to_update = to_update or len(incorrect_transitions) > 0
        if "reward_model" in self.execution_errors:
            log.info(
                f'Iter {iter_num}, step {self.step_num}: Reward model bug: {self.execution_errors["reward_model"]}'
            )
            to_update = True
        else:
            log.info(
                f"Iter {iter_num}, step {self.step_num}: Reward model pass rate: {len(correct_rewards) / (len(correct_rewards) + len(incorrect_rewards))} of {len(correct_rewards) + len(incorrect_rewards)}"
            )
            to_update = to_update or len(incorrect_rewards) > 0

        if "transition_model" in self.execution_errors:
            self.transition_messages.append(
                {"role": "user", "content": self.execution_errors["transition_model"]}
            )
            self.gpt_update_transition_model(iter_num=iter_num)
        elif "reward_model" in self.execution_errors:
            self.reward_messages.append(
                {"role": "user", "content": self.execution_errors["reward_model"]}
            )
            self.gpt_update_reward_model(iter_num=iter_num)
        else:
            self.transition_messages = []
            self.reward_messages = []

            reward_prompt = None
            transition_prompt = None
            if iter_num == 0:
                # First model update. LLM has not generated any code yet
                sampled_transitions = replay_buffer.sample_transitions(7)
                (
                    reward_experiences,
                    transition_experiences,
                ) = get_experience_strings_for_prompts(sampled_transitions)

                reward_prompt, transition_prompt = self.get_starting_prompts(
                    reward_experiences, transition_experiences
                )
            else:
                # Check for cases where the LLM generated models were wrong and feed back
                if len(incorrect_rewards) > 0:
                    reward_prompt = self.get_incorrect_reward_prompt(
                        correct_rewards, incorrect_rewards
                    )

                if len(incorrect_transitions) > 0:
                    transition_prompt = self.get_incorrect_transition_prompt(
                        correct_transitions, incorrect_transitions
                    )

            if reward_prompt is not None:
                self.reward_messages = [{"role": "system", "content": reward_prompt}]
                self.gpt_update_reward_model(iter_num=iter_num)

            if transition_prompt is not None:
                self.transition_messages = [
                    {"role": "system", "content": transition_prompt}
                ]
                self.gpt_update_transition_model(iter_num=iter_num)

        return to_update

    def reset_experiences(self) -> None:
        super().reset()
        self.execution_errors = {}

    def gpt_update_reward_model(self, iter_num: int) -> None:
        code_str, local_scope = requery(
            self.reward_messages,
            REWARD_FUNCTION_NAME,
            iter_num,
            exec_attempt=0,
            replay_path=self.replay_path,
            step_num=self.step_num,
            api=self.code_api,
            use_openrouter=self.use_openrouter,
        )

        if code_str is not None:
            self.current_reward_code = code_str
            assert self.type_tuple is not None
            self.planner.reward_model = reward_model_translator(
                REWARD_FUNCTION_NAME, local_scope, self.type_tuple
            )

    def get_next_action(self, state: StateType) -> ActType:
        if len(self.execution_errors) == 0:
            steps_left = self.max_steps - self.steps_taken
            action, error = self.planner.plan_next_action(state, steps_left)
            self.execution_errors |= error
        else:
            log.info("In error state, returning random action")
            action = random.choice(self.actions)

        self.steps_taken += 1
        return action

    def gpt_update_transition_model(self, iter_num: int) -> None:
        code_str, local_scope = requery(
            self.transition_messages,
            TRANSITION_FUNCTION_NAME,
            iter_num,
            exec_attempt=0,
            replay_path=self.replay_path,
            step_num=self.step_num,
            api=self.code_api,
            use_openrouter=self.use_openrouter,
        )

        if code_str is not None:
            self.current_transition_code = code_str
            assert self.type_tuple is not None
            self.planner.transition_model = transition_model_translator(
                TRANSITION_FUNCTION_NAME, local_scope, self.type_tuple
            )


EMPTY_REWARD_FUNCTION = """
def reward_func(state: MinigridState, action: Actions, next_state: MinigridState) -> Tuple[float, bool]:
    return 0.0, False
"""
EMPTY_TRANSITION_FUNCTION = """
def transition_func(state: MinigridState, action: Actions) -> MinigridState:
    return state
"""
