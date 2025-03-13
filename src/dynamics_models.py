import numpy as np

class PushingDynamicsModel:
    """
    把输入的物体点云（point cloud）沿着给定的夹爪运动方向平移一个指定的距离
    用于 MPC（Model Predictive Control）中，用来预测在某个控制指令下物体点云的变化，以便后续计算代价和选择最佳控制策略

    Heuristics-based pushing dynamics model.
    Translates the object by gripper_moving_distance in gripper_direction.
    """
    def __init__(self):
        pass

    def forward(self, inputs, max_per_batch=2000):
        
        """
        根据输入点云数量 inputs[0].shape[0] 计算需要分成多少批次（num_batch）。
        对每一批次调用内部的 _forward_batched 方法进行预测，然后将所有批次结果拼接在一起。
        返回拼接后的预测点云。
        split inputs into multiple batches if exceeds max_per_batch
        """
        num_batch = int(np.ceil(inputs[0].shape[0] / max_per_batch))
        output = []
        for i in range(num_batch):
            start = i * max_per_batch
            end = (i + 1) * max_per_batch
            output.append(self._forward_batched([x[start:end] for x in inputs]))
        output = np.concatenate(output, axis=0)
        return output

    def _forward_batched(self, inputs):
        '''
        如果夹爪的运动方向与物体中心方向相反，即为无效的“外向推”（outward push）
        将点云沿夹爪运动方向平移相应的距离，从而模拟物体在推送下的位置变化
        '''

        (pcs, contact_position, gripper_direction, gripper_moving_distance) = inputs
        # to float16
        pcs = pcs.astype(np.float16)
        contact_position = contact_position.astype(np.float16)
        gripper_direction = gripper_direction.astype(np.float16)
        gripper_moving_distance = gripper_moving_distance.astype(np.float16)
        # find invalid push (i.e., outward push)
        obj_center = np.mean(pcs, axis=1)  # B x 3
        is_outward = np.sum((obj_center - contact_position) * gripper_direction, axis=1) < 0  # B
        moving_dist = gripper_moving_distance.copy()
        moving_dist[is_outward] = 0
        # translate pc by gripper_moving_distance in gripper_direction
        output = pcs + moving_dist[:, np.newaxis, :] * gripper_direction[:, np.newaxis, :]  # B x N x 3
        return output
