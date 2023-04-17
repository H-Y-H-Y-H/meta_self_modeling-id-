
    # Run a single robot
    if mode == 0:
        taskID = 9
        print('Task:', taskID)
        data_reward_design = 'loop300'
        num_robots_per_task = 200

        Train = 2
        # p.connect(p.GUI) if not Train else p.connect(p.DIRECT)
        p.connect(p.DIRECT)
        para_config = np.loadtxt('../meta_data/para_config.csv')
        all_robot_test = []
        initial_para = para_config[:,0]
        para_range = para_config[:,1:]
        done_times_log = []
        for robotid in range(taskID*num_robots_per_task, (taskID+1)*num_robots_per_task):

            random.seed(2022)
            np.random.seed(2022)

            robot_name = robot_list[robotid]
            if robot_name in kill_list:
                all_robot_test.append(0)
                print(robot_name,'exist')
                continue
            print(robotid,robot_name)

            initial_joints_angle = np.loadtxt(URDF_PTH + "%s/%s.txt" % (robot_name, robot_name))
            initial_joints_angle = initial_joints_angle[0] if len(initial_joints_angle.shape) == 2 else initial_joints_angle
            # Gait parameters and sans data path:
            log_pth = "../meta_data/%s/%s/" % (data_reward_design, robot_name)
            os.makedirs(log_pth, exist_ok=True)

            if Train == 2:
                env = meta_sm_Env(initial_joints_angle, urdf_path=URDF_PTH + "%s/%s.urdf" % (robot_name, robot_name))
                max_train_step = 300
                env.sleep_time = 0
                obs = env.reset()
                step_times = 0
                r_record = -np.inf
                SANS_data = []
                save_action = []
                done_time = 0
                while 1:
                    all_rewards = []
                    action_list = []
                    norm_space = np.random.sample(len(initial_para))
                    action = norm_space * (para_range[:,1]-para_range[:,0]) + para_range[:,0]

                    action_list.append(action)

                    r = 0
                    for i in range(6):
                        step_times += 1
                        next_obs, _, done, _ = env.step(action)
                        sans = np.hstack((obs, action, next_obs))
                        SANS_data.append(sans)
                        obs = next_obs

                        # stable method
                        r += 3 * obs[1] - abs(obs[0]) - abs(obs[2])

                        if done:
                            obs = env.resetBase()
                            done_time += 1
                            break

                    pos, ori = env.robot_location()
                    # 2. forward_1 method
                    # r = pos[1] - abs(pos[0]) - abs(pos[2]) * 0.4
                    # 2. stable method

                    all_rewards.append(r)

                    if r > r_record:
                        r_record = np.copy(r)
                        save_action = np.copy(action)
                    if step_times >= max_train_step:
                        break
                np.savetxt(log_pth + "gait_step%d.csv" % max_train_step, np.asarray(save_action))
                print("step count:", step_times, "r:", r_record)
                done_times_log.append(done_time)
                np.savetxt(log_pth + "sans_%d.csv" % max_train_step, np.asarray(SANS_data))
                # p.disconnect()


            else:
                env = meta_sm_Env(initial_joints_angle, urdf_path=URDF_PTH + "%s/%s.urdf" % (robot_name, robot_name))
                action = np.loadtxt(log_pth + '/gait_step300.csv')

                env.sleep_time = 0
                obs = env.reset()

                for epoch_run in range(1):
                    obs = env.resetBase()
                    r_log = 0
                    for i in range(6):
                        next_obs, r, done, _ = env.step(action)
                        r_log += r
                        if done:
                            break
                    all_robot_test.append(r_log)
        np.savetxt('../meta_data/test_logger2000_%d.csv'%taskID, np.asarray(all_robot_test))
        # np.savetxt('../meta_data/done_logger2000_%d.csv'%taskID, np.asarray(done_times_log))
