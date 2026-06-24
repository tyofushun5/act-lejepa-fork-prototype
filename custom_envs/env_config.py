from pydantic import BaseModel


class MetaworldEnvConfig(BaseModel):

    @staticmethod
    def env_tasks():
        '''Return a DataFrame with id, env_name, and description.'''
        import pandas as pd
        data = [
            {'id': 0,  'env_name': 'assembly-v3', 'description': 'Pick up a nut and place it onto a peg.'},
            {'id': 1,  'env_name': 'basketball-v3', 'description': 'Dunk the basketball into the basket.'},
            {'id': 2,  'env_name': 'bin-picking-v3', 'description': 'Grasp the puck from one bin and place it into another bin.'},
            {'id': 3,  'env_name': 'box-close-v3', 'description': 'Grasp the cover and close the box with it.'},
            {'id': 4,  'env_name': 'button-press-topdown-v3', 'description': 'Press a button from the top.'},
            {'id': 5,  'env_name': 'button-press-topdown-wall-v3', 'description': 'Bypass a wall and press a button from the top.'},
            {'id': 6,  'env_name': 'button-press-v3', 'description': 'Press a button.'},
            {'id': 7,  'env_name': 'button-press-wall-v3', 'description': 'Bypass a wall and press a button.'},
            {'id': 8,  'env_name': 'coffee-button-v3', 'description': 'Push a button on the coffee machine.'},
            {'id': 9,  'env_name': 'coffee-pull-v3', 'description': 'Pull a mug from a coffee machine.'},
            {'id': 10, 'env_name': 'coffee-push-v3', 'description': 'Push a mug under a coffee machine.'},
            {'id': 11, 'env_name': 'dial-turn-v3', 'description': 'Rotate a dial 180 degrees.'},
            {'id': 12, 'env_name': 'disassemble-v3', 'description': 'Pick a nut out of the a peg.'},
            {'id': 13, 'env_name': 'door-close-v3', 'description': 'Close a door with a revolving joint.'},
            {'id': 14, 'env_name': 'door-lock-v3', 'description': 'Lock the door by rotating the lock clockwise.'},
            {'id': 15, 'env_name': 'door-open-v3', 'description': 'Open a door with a revolving joint.'},
            {'id': 16, 'env_name': 'door-unlock-v3', 'description': 'Unlock the door by rotating the lock counter-clockwise.'},
            {'id': 17, 'env_name': 'hand-insert-v3', 'description': 'Insert the gripper into a hole.'},
            {'id': 18, 'env_name': 'drawer-close-v3', 'description': 'Push and close a drawer.'},
            {'id': 19, 'env_name': 'drawer-open-v3', 'description': 'Open a drawer.'},
            {'id': 20, 'env_name': 'faucet-open-v3', 'description': 'Rotate the faucet counter-clockwise.'},
            {'id': 21, 'env_name': 'faucet-close-v3', 'description': 'Rotate the faucet clockwise.'},
            {'id': 22, 'env_name': 'hammer-v3', 'description': 'Hammer a screw on the wall.'},
            {'id': 23, 'env_name': 'handle-press-side-v3', 'description': 'Press a handle down sideways.'},
            {'id': 24, 'env_name': 'handle-press-v3', 'description': 'Press a handle down.'},
            {'id': 25, 'env_name': 'handle-pull-side-v3', 'description': 'Pull a handle up sideways.'},
            {'id': 26, 'env_name': 'handle-pull-v3', 'description': 'Pull a handle up.'},
            {'id': 27, 'env_name': 'lever-pull-v3', 'description': 'Pull a lever down 90 degrees.'},
            {'id': 28, 'env_name': 'pick-place-wall-v3', 'description': 'Pick a puck, bypass a wall and place the puck.'},
            {'id': 29, 'env_name': 'pick-out-of-hole-v3', 'description': 'Pick up a puck from a hole.'},
            {'id': 30, 'env_name': 'pick-place-v3', 'description': 'Pick and place a puck to a goal.'},
            {'id': 31, 'env_name': 'plate-slide-v3', 'description': 'Slide a plate into a cabinet.'},
            {'id': 32, 'env_name': 'plate-slide-side-v3', 'description': 'Slide a plate into a cabinet sideways.'},
            {'id': 33, 'env_name': 'plate-slide-back-v3', 'description': 'Get a plate from the cabinet.'},
            {'id': 34, 'env_name': 'plate-slide-back-side-v3', 'description': 'Get a plate from the cabinet sideways.'},
            {'id': 35, 'env_name': 'peg-insert-side-v3', 'description': 'Insert a peg sideways.'},
            {'id': 36, 'env_name': 'peg-unplug-side-v3', 'description': 'Unplug a peg sideways.'},
            {'id': 37, 'env_name': 'soccer-v3', 'description': 'Kick a soccer into the goal.'},
            {'id': 38, 'env_name': 'stick-push-v3', 'description': 'Grasp a stick and push a box using the stick.'},
            {'id': 39, 'env_name': 'stick-pull-v3', 'description': 'Grasp a stick and pull a box with the stick.'},
            {'id': 40, 'env_name': 'push-v3', 'description': 'Push the puck to a goal.'},
            {'id': 41, 'env_name': 'push-wall-v3', 'description': 'Bypass a wall and push a puck to a goal.'},
            {'id': 42, 'env_name': 'push-back-v3', 'description': 'Pull a puck to a goal.'},
            {'id': 43, 'env_name': 'reach-v3', 'description': 'Reach a goal position.'},
            {'id': 44, 'env_name': 'reach-wall-v3', 'description': 'Bypass a wall and reach a goal.'},
            {'id': 45, 'env_name': 'shelf-place-v3', 'description': 'Pick and place a puck onto a shelf.'},
            {'id': 46, 'env_name': 'sweep-into-v3', 'description': 'Sweep a puck into a hole.'},
            {'id': 47, 'env_name': 'sweep-v3', 'description': 'Sweep a puck off the table.'},
            {'id': 48, 'env_name': 'window-open-v3', 'description': 'Push and open a window.'},
            {'id': 49, 'env_name': 'window-close-v3', 'description': 'Push and close a window.'}
        ]
        return pd.DataFrame(data)
    

class ManiSkillEnvConfig(BaseModel):

    @staticmethod
    def env_tasks():
        '''Return a DataFrame with id, env_name, and description.'''
        import pandas as pd
        data = [
            # {'id': 0,  'env_name': 'LiftPegUpright-v1', 'description': ''},
            {'id': 1,  'env_name': 'PegInsertionSide-v1', 'description': ''},
            {'id': 2,  'env_name': 'PickCube-v1', 'description': ''},
            # {'id': 3,  'env_name': 'PokeCube-v1', 'description': ''},
            # {'id': 4,  'env_name': 'PullCube-v1', 'description': ''},
            {'id': 5,  'env_name': 'PullCubeTool-v1', 'description': ''},
            {'id': 6,  'env_name': 'PushCube-v1', 'description': ''},
            # {'id': 7, 'env_name': 'RollBall-v1', 'description': ''},
            {'id': 8, 'env_name': 'StackCube-v1', 'description': ''},
            # {'id': 7, 'env_name': 'PushT-v1', 'description': ''},
            # {'id': 10,  'env_name': 'AnymalC-Reach-v1', 'description': ''},
            # {'id': 11,  'env_name': 'DrawTriangle-v1', 'description': ''},
            # {'id': 12, 'env_name': 'StackPyramid-v1', 'description': ''},
            # {'id': 13, 'env_name': 'TwoRobotPickCube-v1', 'description': ''},
            # {'id': 14, 'env_name': 'TwoRobotStackCube-v1', 'description': ''},
            # {'id': 15,  'env_name': 'PlugCharger-v1', 'description': ''},
        ]
        return pd.DataFrame(data)
    