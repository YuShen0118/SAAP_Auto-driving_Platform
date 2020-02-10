using System;
using System.Collections.Generic;
using UnityEngine;
using UnityStandardAssets.CrossPlatformInput;
using System.IO;
using RVO;


using Vector2 = RVO.Vector2;
using Random = System.Random;

namespace UnityStandardAssets.Vehicles.Car
{
    [RequireComponent(typeof(CarController))]
    public class CarRemoteControl : MonoBehaviour
    {
        public bool useExpert = true;
        private CarController m_Car; // the car controller we want to use

        public float SteeringAngle { get; set; }
        public float Acceleration { get; set; }
        public bool UpdatedFlag {get; set;}
        private Steering s;


        //float RVO_scale = 1.0f;
        int frameIdx = 0;
        GameObject mainCar;
        IList<Vector2> goals;
        float preferred_speed = 10.0f;

        List<IList<Vector2>> obstacles = new List<IList<Vector2>>();

        List<Vector3> checkpoints = new List<Vector3>();
        List<GameObject> checkpointObjs = new List<GameObject>();
        /** Random number generator. */
        Random random;

        //tmp
        //double pre_direction = 0;


        private void Awake()
        {
            // get the car controller
            m_Car = GetComponent<CarController>();
            s = new Steering();
            s.Start();

            initPlanningStates();
        }

        void initPlanningStates()
        {
            UpdatedFlag = false;
            mainCar = GameObject.Find("MainCar");
            float y = 5;
            //original one
            //             checkpoints.Add(new Vector3(72.0f, y, 10.0f));
            //             checkpoints.Add(new Vector3(70.0f, y, 20.0f));
            //             checkpoints.Add(new Vector3(70.0f, y, 30.0f));
            //             checkpoints.Add(new Vector3(60.0f, y, 38.0f));
            //             checkpoints.Add(new Vector3(50.0f, y, 43.0f));
            //             checkpoints.Add(new Vector3(0.0f, y, 45.0f));

            checkpoints.Add(new Vector3(50.0f, y, 44.0f));
            checkpoints.Add(new Vector3(-47.0f, y, 46.0f));
            checkpoints.Add(new Vector3(-52.0f, y, -45.0f));
            checkpoints.Add(new Vector3(-10.0f, y, -66.0f));
            checkpoints.Add(new Vector3(-3.0f, y, 42.0f));

            for (int i = 0; i < checkpoints.Count; i++)
            {
                GameObject obj = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                obj.transform.position = checkpoints[i];
                checkpointObjs.Add(obj);
            }


            IList<Vector2> obstacle = new List<Vector2>();
            obstacle.Add(new Vector2(84, 60));
            obstacle.Add(new Vector2(-49, 60));
            obstacle.Add(new Vector2(-49, 49));
            obstacle.Add(new Vector2(14, 49));
            obstacle.Add(new Vector2(28, 48));
            obstacle.Add(new Vector2(62, 48));
            obstacle.Add(new Vector2(81, 23));
            obstacle.Add(new Vector2(84, 0));
            obstacles.Add(obstacle);

            obstacle = new List<Vector2>();
            obstacle.Add(new Vector2(70, 0));
            obstacle.Add(new Vector2(69.5f, 14));
            obstacle.Add(new Vector2(67.5f, 20));
            obstacle.Add(new Vector2(64, 27));
            obstacle.Add(new Vector2(55, 36));
            obstacle.Add(new Vector2(40, 42));
            obstacle.Add(new Vector2(0, 42));
            obstacle.Add(new Vector2(0, -22));
            obstacle.Add(new Vector2(7, -35));
            obstacle.Add(new Vector2(7, -56));
            obstacle.Add(new Vector2(70, -56));
            obstacles.Add(obstacle);

            obstacle = new List<Vector2>();
            obstacle.Add(new Vector2(-7, 42));
            obstacle.Add(new Vector2(-49, 42));
            obstacle.Add(new Vector2(-49, -16));
            obstacle.Add(new Vector2(-42, -28));
            obstacle.Add(new Vector2(-42, -45));
            obstacle.Add(new Vector2(-37, -53));
            obstacle.Add(new Vector2(-28, -56));
            obstacle.Add(new Vector2(-7, -56));
            obstacles.Add(obstacle);

            obstacle = new List<Vector2>();
            obstacle.Add(new Vector2(-56, -70));
            obstacle.Add(new Vector2(84, -70));
            obstacle.Add(new Vector2(-28, -70));
            obstacle.Add(new Vector2(-47, -63));
            obstacle.Add(new Vector2(-56, -45));
            obstacle.Add(new Vector2(-56, 60));
            obstacles.Add(obstacle);

            //drawMap();
        }

        void drawMap()
        {
//             Graphics g;
// 
//             g = this.CreateGraphics();
// 
//             Pen myPen = new Pen(Color.Red);
//             myPen.Width = 30;
//             g.DrawLine(myPen, 30, 30, 45, 65);
// 
//             g.DrawLine(myPen, 1, 1, 45, 65);
        }

        bool reachGoal(double reachRange = 2)
        {
            Vector3 projection = new Vector3(checkpoints[0].x, mainCar.transform.position.y, checkpoints[0].z);
            if ((projection - mainCar.transform.position).magnitude < reachRange)
            {
                return true;
            }

            return false;
        }

        void updateCheckpoints()
        {
            if (checkpoints.Count > 1 && reachGoal())
            {
                checkpointObjs[0].transform.position = new Vector3(checkpoints[0].x, checkpoints[0].y - 10, checkpoints[0].z);
                checkpointObjs.RemoveAt(0);
                checkpoints.RemoveAt(0);
            }
        }

        void drive()
        {
            goals = new List<Vector2>();

            updateCheckpoints();

#if RVO_SEED_RANDOM_NUMBER_GENERATOR
        random = new Random();
#else
            random = new System.Random(0);
#endif
            setupScenario();

            setPreferredVelocities();
            
            Simulator.Instance.doStep();

            //             mainCar.transform.position = new Vector3(pos_new.x(), mainCar.transform.position.y, pos_new.y());

            calculateVandS();

            Simulator.Instance.Clear();
        }

        void calculateVandS()
        {
            Vector2 pos_new = Simulator.Instance.getAgentPosition(0);

            double car_direction = mainCar.transform.eulerAngles.y / 180.0 * Math.PI;
            
            double dx = pos_new.x() - mainCar.transform.position.x;
            double dz = pos_new.y() - mainCar.transform.position.z;
            double target_direction = -Math.Atan2(dz, dx) + Math.PI / 2;

            double delta_direction = target_direction - car_direction;
            while (delta_direction > Math.PI) delta_direction -= Math.PI * 2;
            while (delta_direction < -Math.PI) delta_direction += Math.PI * 2;

            SteeringAngle = (float)(delta_direction / Math.PI * 180.0 / m_Car.MaxSteerAngleInDegree);

//             double dt_direction = car_direction - pre_direction;
//             pre_direction = car_direction;
//             Debug.Log("dt_direction " + (dt_direction * 50 / Math.PI * 180.0).ToString());
//             Debug.Log("delta_direction " + delta_direction.ToString());
//             Debug.Log("SteeringAngle " + (SteeringAngle * m_Car.MaxSteerAngleInDegree).ToString());
            SteeringAngle = Mathf.Clamp(SteeringAngle, -1, 1);

            float current_speed = m_Car.CurrentSpeed;
            var desired_velocity = Simulator.Instance.getAgentVelocity(0);
            Vector3 des_v = new Vector3(desired_velocity.x(), 0, desired_velocity.y());
            float desired_speed = des_v.magnitude;

            //             Debug.Log("pos_new " + pos_new.ToString());
            //             Debug.Log("current_speed " + current_speed.ToString());
            //             Debug.Log("desired_speed " + desired_speed.ToString());
            // 
            //             double zz = Math.Sqrt( dx * dx + dz * dz) * 50;
            //             Debug.Log("zz " + zz.ToString());
            
            Acceleration = (desired_speed - current_speed) / 0.02f;
            if (Acceleration > 1) Acceleration = 1;
            if (Acceleration < 0)
            {
                if (desired_speed - current_speed > -1) Acceleration = 0;
            }
            //Debug.Log("frameIdx " + frameIdx.ToString() + " Acceleration " + Acceleration.ToString());

            frameIdx++;
            //if (frameIdx > 100) Acceleration = 0;

        }

        bool reachedGoal()
        {
            /* Check if all agents have reached their goals. */
            for (int i = 0; i < Simulator.Instance.getNumAgents(); ++i)
            {
                if (RVOMath.absSq(Simulator.Instance.getAgentPosition(i) - goals[i]) > 400.0f)
                {
                    return false;
                }
            }

            return true;
        }


        void setPreferredVelocities()
        {
            /*
                * Set the preferred velocity to be a vector of unit magnitude
                * (speed) in the direction of the goal.
                */
            for (int i = 0; i < Simulator.Instance.getNumAgents(); ++i)
            {
                Vector2 goalVector = goals[i] - Simulator.Instance.getAgentPosition(i);

                if (RVOMath.absSq(goalVector) > preferred_speed)
                {
                    goalVector = RVOMath.normalize(goalVector) * preferred_speed;
                }

                Simulator.Instance.setAgentPrefVelocity(i, goalVector);

                /* Perturb a little to avoid deadlocks due to perfect symmetry. */
                float angle = (float)random.NextDouble() * 2.0f * (float)Math.PI;
                float dist = (float)random.NextDouble() * 0.0001f;

                Simulator.Instance.setAgentPrefVelocity(i, Simulator.Instance.getAgentPrefVelocity(i) +
                    dist * new Vector2((float)Math.Cos(angle), (float)Math.Sin(angle)));
            }
        }

        void setupScenario()
        {
            /* Specify the global time step of the simulation. */
            Simulator.Instance.setTimeStep(0.02f);

            /*
                * Specify the default parameters for agents that are subsequently
                * added.
                */
            Simulator.Instance.setAgentDefaults(7.0f, 10, 1.0f, 0.1f, 1.5f, 20.0f, new Vector2(0.0f, 0.0f));
            
            var position = mainCar.transform.position;
            Simulator.Instance.addAgent(new Vector2(position.x, position.z));
            goals.Add(new Vector2(checkpoints[0].x, checkpoints[0].z));

            for (int i=0; i<obstacles.Count; i++)//obstacles.Count
            {
                Simulator.Instance.addObstacle(obstacles[i]);
            }
            Simulator.Instance.processObstacles();
            
// 
//             /*
//                 * Add (polygonal) obstacles, specifying their vertices in
//                 * counterclockwise order.
//                 */
//             IList<Vector2> obstacle1 = new List<Vector2>();
//             obstacle1.Add(new Vector2(-10.0f, 40.0f));
//             obstacle1.Add(new Vector2(-40.0f, 40.0f));
//             obstacle1.Add(new Vector2(-40.0f, 10.0f));
//             obstacle1.Add(new Vector2(-10.0f, 10.0f));
//             Simulator.Instance.addObstacle(obstacle1);
// 
//             IList<Vector2> obstacle2 = new List<Vector2>();
//             obstacle2.Add(new Vector2(10.0f, 40.0f));
//             obstacle2.Add(new Vector2(10.0f, 10.0f));
//             obstacle2.Add(new Vector2(40.0f, 10.0f));
//             obstacle2.Add(new Vector2(40.0f, 40.0f));
//             Simulator.Instance.addObstacle(obstacle2);
// 
//             IList<Vector2> obstacle3 = new List<Vector2>();
//             obstacle3.Add(new Vector2(10.0f, -40.0f));
//             obstacle3.Add(new Vector2(40.0f, -40.0f));
//             obstacle3.Add(new Vector2(40.0f, -10.0f));
//             obstacle3.Add(new Vector2(10.0f, -10.0f));
//             Simulator.Instance.addObstacle(obstacle3);
// 
//             IList<Vector2> obstacle4 = new List<Vector2>();
//             obstacle4.Add(new Vector2(-10.0f, -40.0f));
//             obstacle4.Add(new Vector2(-10.0f, -10.0f));
//             obstacle4.Add(new Vector2(-40.0f, -10.0f));
//             obstacle4.Add(new Vector2(-40.0f, -40.0f));
//             Simulator.Instance.addObstacle(obstacle4);
// 
//             /*
//                 * Process the obstacles so that they are accounted for in the
//                 * simulation.
//                 */
//             Simulator.Instance.processObstacles();
        }

        Bounds GetMaxBounds(GameObject g)
        {
            var b = new Bounds(g.transform.position, Vector3.zero);
            foreach (Renderer r in g.GetComponentsInChildren<Renderer>())
            {
                b.Encapsulate(r.bounds);
            }
            return b;
        }

        private void FixedUpdate()
        {
            Bounds bbx = GetMaxBounds(mainCar);

            if (useExpert)
            {
                drive();
            }

            // If holding down W or S control the car manually
            if (Input.GetKey(KeyCode.W) || Input.GetKey(KeyCode.S) || Input.GetKey(KeyCode.A) || Input.GetKey(KeyCode.D))
            {
                s.UpdateValues();
                //Debug.Log("s.v " + s.V);
                m_Car.Move(s.H, s.V, s.V, 0f);

            }
            else
            {
//                 Debug.Log("m_Car.CurrentSpeed " + m_Car.CurrentSpeed.ToString());
//                 Debug.Log("Acceleration " + Acceleration.ToString());
//                 Debug.Log("SteeringAngle " + SteeringAngle.ToString());
                if (UpdatedFlag || useExpert)
                {
                    UpdatedFlag = false;
                    m_Car.Move(SteeringAngle, Acceleration, Acceleration, 0f);
                }
            }
        }
    }
}
