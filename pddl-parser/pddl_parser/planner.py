#!/usr/bin/env python
# Four spaces as indentation [no tabs]

# This file is part of PDDL Parser, available at <https://github.com/pucrs-automated-planning/pddl-parser>.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>


from .PDDL import PDDL_Parser
import time
import signal


def timeout_handler(signum, frame):
    raise TimeoutError("solve() 执行超时，被强行中断！")

class Planner:

    def applicable(self, state, positive, negative):
        # 这里填入你的条件判断代码
        return positive.issubset(state) and negative.isdisjoint(state)

    def apply(self, state, add_effects, del_effects):
        # 这里填入你的状态更新逻辑
        return state.difference(del_effects).union(add_effects)

    def solve(self, domain, problem, state=None):
        # 注册信号处理函数，并设定超时时间（例如 20 秒）
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(10)  # 20 秒后触发 SIGALRM 信号

        try:
            # Parser 部分
            parser = PDDL_Parser()
            parser.parse_domain(domain)
            parser.parse_problem(problem)
            if state is None:
                state = parser.state
            # 目标条件
            goal_pos = parser.positive_goals
            goal_not = parser.negative_goals

            # 若当前状态已满足目标，返回空计划
            if self.applicable(state, goal_pos, goal_not):
                return []

            # Grounding process：生成所有地面化动作
            ground_actions = []
            for action in parser.actions:
                for act in action.groundify(parser.objects, parser.types):
                    ground_actions.append(act)

            # Search 部分
            visited = set([state])
            fringe = [state, None]
            while fringe:
                state = fringe.pop(0)
                plan = fringe.pop(0)
                for act in ground_actions:
                    if self.applicable(state, act.positive_preconditions, act.negative_preconditions):
                        new_state = self.apply(state, act.add_effects, act.del_effects)
                        if new_state not in visited:
                            if self.applicable(new_state, goal_pos, goal_not):
                                full_plan = [act]
                                while plan:
                                    act, plan = plan
                                    full_plan.insert(0, act)
                                return full_plan
                            visited.add(new_state)
                            fringe.append(new_state)
                            fringe.append((act, plan))
            return None

        except (TimeoutError, TypeError, AttributeError, ValueError, IndexError, KeyError) as e:
            # print(domain)
            # print(e)
            return e

        finally:
            # 取消定时器，防止影响后续代码
            signal.alarm(0)



# -----------------------------------------------
# Main
# -----------------------------------------------
if __name__ == '__main__':
    import sys, time
    start_time = time.time()
    domain = sys.argv[1]
    problem = sys.argv[2]
    verbose = len(sys.argv) > 3 and sys.argv[3] == '-v'
    planner = Planner()
    plan = planner.solve(domain, problem)
    print('Time: ' + str(time.time() - start_time) + 's')
    if plan is not None:
        print('plan:')
        for act in plan:
            print(act if verbose else act.name + ' ' + ' '.join(act.parameters))
    else:
        sys.exit('No plan was found')
