# 碰撞检测

默认构建提供基于 Conan 管理的 Pinocchio 和 Coal 的 `CollisionModel`。它从调用方
传入的 URDF 加载碰撞几何，并检测指定构型下的自碰撞。

可以使用 `add_box_obstacle` 动态加入世界坐标系中的 box，并通过
`remove_obstacle` 删除。把障碍物名称加入碰撞组即可选择需要与其检测的机器人
link。

```bash
./scripts/build.sh
python3 examples/python/collision/basic_query.py \
  --urdf /绝对路径/robot.urdf \
  -q 0 0 0 0 0 0
```

`-q` 后的数值数量必须等于程序输出的 Pinocchio `nq`。固定基座机器人默认使用
全零构型；浮动基座机器人应显式传入合法的 Pinocchio 构型，其中四元数必须归一化。

如果 URDF 使用 `package://` 网格路径，可重复传入 package 根目录：

```bash
python3 examples/python/collision/basic_query.py \
  --urdf /path/to/robot.urdf \
  --package-dir /path/to/ros/workspace/src \
  -q 0 0 0 0 0 0
```

默认排除相邻连杆的碰撞对；需要检测它们时加入 `--include-adjacent`。使用
`--help` 查看全部参数。

## 碰撞对分组管理

对于双臂机器人，可以把左右臂分别定义成 link 组，并且只启用左右臂组间的
碰撞对。挂载在所列 link 上的所有碰撞几何体都会归入对应机械臂：

```bash
python3 examples/python/collision/basic_query.py --urdf /path/to/dual_arm.urdf \
  --group left_arm=left_shoulder,left_upper_arm,left_forearm,left_hand \
  --group right_arm=right_shoulder,right_upper_arm,right_forearm,right_hand \
  --check-groups left_arm:right_arm \
  -q 0 0 0 0 0 0 0 0 0 0 0 0
```

这会检测所有左臂几何体与所有右臂几何体，但不检测单臂内部。加入
`--check-groups left_arm:left_arm` 或 `right_arm:right_arm` 即可开启对应机械臂
内部的自碰撞检测。
组内自碰撞会自动排除同一 link 和相邻 link 的碰撞对。

对应的 Python API：

```python
model.set_collision_groups(
    {"left_arm": left_links, "right_arm": right_links},
    [("left_arm", "right_arm")],
)
for pair in model.collision_pairs:
    print(pair.first_link, pair.second_link)
```

`set_collision_groups()` 会替换当前活动碰撞对。调用 `reset_collision_pairs()`
恢复全部碰撞对，默认仍排除相邻连杆。也可以继续使用
`remove_collision_pair(first_geometry, second_geometry)` 删除单个几何碰撞对。

## 碰撞球后端

`SphereCollisionModel` 面向高频构型和轨迹查询。调用方显式提供 link 局部坐标系
下的正半径球；Pinocchio 在 FK 时更新世界球心，随后只计算活动球对。接口支持
语义分组、安全距离、有符号最小距离、世界坐标球输出和 `[batch, nq]` 批量查询。

```python
spheres = [
    hm.CollisionSphere("left_upper_0", "left_upper_arm", [0, 0, 0.12], 0.09),
    hm.CollisionSphere("right_upper_0", "right_upper_arm", [0, 0, 0.12], 0.09),
]
sphere_model = hm.SphereCollisionModel(urdf_path, spheres)
sphere_model.set_collision_groups(
    {"left_arm": left_links, "right_arm": right_links},
    [("left_arm", "right_arm")],
)
distance = sphere_model.minimum_distance(q).distance
trajectory_distances = sphere_model.batch_minimum_distances(q_path)

planner = hm.SamplingPlanner.from_sphere_collision_model(
    lower_limits, upper_limits, sphere_model, security_margin=0.01
)
```

球模型属于近似几何，并且库不会隐式生成球。快速拒绝查询应使用保守膨胀且经过
检查的球模型；除非已经单独证明球集合的覆盖性，否则最终轨迹仍应交给 Coal 做
精确验证。每个实例持有可变的 Pinocchio 查询缓冲区，并发查询时需要调用方同步。

该设计吸收了 cuRobo 碰撞球表示和批量查询架构的思想，但围绕 Pinocchio 与
Eigen 独立实现；HolisticMotion 不导入 Torch、Warp 或 cuRobo。

### 离线球化

NumPy 拟合器根据内部采样点与表面采样点选择较大的中轴候选球；网格适配层使用
可选 `trimesh` 进行体素化。`inscribed` 模式保留采样意义下的内接半径；可选的
`sampled_coverage` 模式会扩张选中的球，使所有输入样本被覆盖。后者仍是离散近似，
不能当作连续三角网格覆盖的数学证明。

```bash
python -m pip install '.[examples]'
./scripts/run.sh python3 \
  examples/python/visualization/sphere_fit_viser.py \
  --mesh /absolute/path/to/link.stl \
  --link left_forearm \
  --max-spheres 24 \
  --sampled-coverage \
  --padding 0.003 \
  --output left_forearm.spheres.json
```

无 Viewer 的构建机器可加入 `--fit-only`。输出 JSON 带格式版本并保存 link 局部
坐标球；使用 `load_sphere_model()` 加载，再通过 `make_collision_spheres()` 转换后
构造 `SphereCollisionModel`。用于规划前，应同时检查 sampled coverage、平均 gap、
最大 gap 和 Viser 叠加效果。

整机批处理工具会读取全部 URDF `<collision>` 元素，并应用 mesh scale 与 collision
origin。支持 mesh、box、cylinder 和 sphere；`package://` 资源可重复传入
`--package-dir`，只拟合部分 link 时可重复传入 `--link`：

```bash
./scripts/run.sh python3 examples/python/collision/fit_urdf_spheres.py \
  --urdf /absolute/path/to/robot.urdf \
  --package-dir /path/to/ros/src \
  --sampled-coverage --padding 0.003 \
  --output robot.spheres.json
```

轻量编辑器在浏览器中提供 link 选择、球数、体素 pitch、最小半径、padding、覆盖
模式、重新拟合、指标显示和保存：

```bash
./scripts/run.sh python3 \
  examples/python/visualization/sphere_model_editor.py \
  --urdf /absolute/path/to/robot.urdf \
  --output robot.spheres.json
```

如果输出 JSON 已存在，编辑器会加载并显示已有球。重新拟合单个 link 时会保留所有
未改动 link；需要明确执行整机重新生成时可使用 **Fit all links**。保存使用原子
替换，校验或写入失败不会留下半写入的模型文件。

## HumanoidAssets Gizmo Demo

双臂 Viser demo 要求显式传入 Marvin URDF 或资产根目录。拖动任一末端 Gizmo 后，
程序会求解 IK，并实时执行左右臂检测。完整机械臂组包含运动链以及
EE 的所有带碰撞几何后代，因此 hand base、手指、tool 和夹具 link 都会参与：

```bash
python -m pip install '.[examples]'
./scripts/build.sh
python3 examples/python/visualization/dual_arm_collision_gizmo.py \
  --urdf /path/to/robot_with_ee.urdf
```

Collision 面板显示安全/碰撞状态和最小距离；红色标记与连线显示 Coal 返回的两个
最近点。常用变体：

**Collision mode** 可以直接切换“完整双臂（含夹具）互检”“仅机械臂链互检”
“完整双臂互检并检测组内自碰撞”和“关闭”。**Active collision pair** 展示展开后的
geometry 碰撞对；可用 **Disable selected pair** 临时禁用，用
**Restore mode pairs** 按当前模式恢复。

通过 **Safety margin (mm)** 设置安全距离，并开启 **Reject unsafe IK**，即可在
候选解低于阈值时保持上一个安全姿态。Marvin demo 默认过滤腕部机构已知的
`wrist_pitch_j5`/`wrist_roll_j7` 对；其他语义禁配对可通过
`--disable-link-pair LINK1:LINK2` 添加。

```bash
# 同时检测左右臂各自内部的自碰撞。
python3 examples/python/visualization/dual_arm_collision_gizmo.py --arm-self-collision

# 验证 IK、资源加载、组展开以及一次距离查询。
python3 examples/python/visualization/dual_arm_collision_gizmo.py --validate-only

# 测量“碰撞 + 最小距离”组合查询性能。
python3 examples/python/visualization/dual_arm_collision_gizmo.py \
  --validate-only --benchmark-samples 200
```

当前开发机实测：含夹具的 Marvin 完整双臂互检展开为 100 对，平均约
40.23 ms，约 25 Hz；开启过滤后的组内自碰撞共 164 对，平均约 87.98 ms，
约 11 Hz。如果只需要机械臂本体，可在 Viser 中选择 **Between arm chains only**，
使用更快的 49 对检测。结果会随 CPU、网格、构型及 Coal broad phase 状态变化，
部署时应在目标机器重新测试。

### 实时查询等级

Gizmo 不再每帧计算精确距离，而是使用两级查询：

- `is_within_distance(q, margin)` 使用 Coal collision request 的安全边界和
  early stop，每帧进行安全判断。
- `evaluate(q)` 返回全部碰撞、精确最小距离和最近点，按照界面中的
  **Exact distance rate (Hz)** 降频执行。

对于包含 100 个碰撞对的 Marvin 完整双臂，开发机上的 20 mm 阈值查询平均约
0.0076 ms、P95 约 0.0077 ms，而精确距离查询仍约 40 ms。精确距离默认以 4 Hz
刷新，但每次有效 Gizmo 更新仍会执行快速安全判断。

其他命令行示例：

- `collision_demo.py`：检测单个构型。
- `collision_pair_management_demo.py`：查看、删除和恢复碰撞对。
- `dual_arm_collision_demo.py`：无 GUI 的双臂语义分组检测。
- `collision_path_scan_demo.py`：沿关节空间线段采样检测碰撞。
