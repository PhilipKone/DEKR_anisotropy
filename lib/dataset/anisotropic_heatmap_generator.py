import numpy as np

class AnisotropicHeatmapGenerator:
    def __init__(self, output_res, num_joints, use_jnt=True, jnt_thr=0.01, use_int=True, shape=False, shape_weight=1.0, pauta=3):
        self.output_res = output_res
        self.num_joints = num_joints
        self.use_jnt = use_jnt
        self.jnt_thr = jnt_thr
        self.use_int = use_int
        self.shape = shape
        self.shape_weight = shape_weight
        self.pauta = pauta

    def get_heat_val(self, sigma_x, sigma_y, x, y, x0, y0, theta=0):
        """
        Oriented anisotropic Gaussian kernel.
        theta: orientation angle in radians (counterclockwise from x-axis)
        """
        dx = x - x0
        dy = y - y0
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        x_rot = cos_t * dx + sin_t * dy
        y_rot = -sin_t * dx + cos_t * dy
        return np.exp(- ((x_rot ** 2) / (2 * sigma_x ** 2) + (y_rot ** 2) / (2 * sigma_y ** 2)))

    def __call__(self, joints, sigmas, ct_sigma, bg_weight=1.0, orientations=None):
        hms = np.zeros((self.num_joints, self.output_res, self.output_res), dtype=np.float32)
        ignored_hms = 2 * np.ones((1, self.output_res, self.output_res), dtype=np.float32)
        hms_list = [hms, ignored_hms]
        for p_idx, p in enumerate(joints):
            for idx, pt in enumerate(p):
                if idx < self.num_joints - 1:
                    sigma = pt[3] if len(pt) > 3 else sigmas[idx]
                else:
                    sigma = ct_sigma
                if pt[2] > 0:
                    x, y = pt[0], pt[1]
                    if x < 0 or y < 0 or x >= self.output_res or y >= self.output_res:
                        continue
                    theta = 0
                    if orientations is not None:
                        theta = orientations[p_idx, idx] if orientations.shape == (joints.shape[0], self.num_joints) else orientations[idx]
                    if self.use_jnt:
                        radius_x = np.sqrt(np.log(1 / self.jnt_thr) * 2 * sigma ** 2)
                        radius_y = 2 * radius_x  # Example: anisotropic, y spread is 2x x spread
                        if self.use_int:
                            radius_x = int(np.floor(radius_x))
                            radius_y = int(np.floor(radius_y))
                        ul = int(np.floor(x - radius_x - 1)), int(np.floor(y - radius_y - 1))
                        br = int(np.ceil(x + radius_x + 2)), int(np.ceil(y + radius_y + 2))
                    else:
                        ul = int(np.floor(x - self.pauta * sigma - 1)), int(np.floor(y - self.pauta * sigma - 1))
                        br = int(np.ceil(x + self.pauta * sigma + 2)), int(np.ceil(y + self.pauta * sigma + 2))
                    cc, dd = max(0, ul[0]), min(br[0], self.output_res)
                    aa, bb = max(0, ul[1]), min(br[1], self.output_res)
                    joint_rg = np.zeros((bb - aa, dd - cc))
                    for sy in range(aa, bb):
                        for sx in range(cc, dd):
                            joint_rg[sy - aa, sx - cc] = self.get_heat_val(sigma, 2 * sigma, sx, sy, x, y, theta)
                    hms_list[0][idx, aa:bb, cc:dd] = np.maximum(hms_list[0][idx, aa:bb, cc:dd], joint_rg)
                    hms_list[1][0, aa:bb, cc:dd] = 1.
        hms_list[1][hms_list[1] == 2] = bg_weight
        return hms_list
