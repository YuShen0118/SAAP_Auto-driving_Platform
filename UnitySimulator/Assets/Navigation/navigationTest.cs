#define RVO_OUTPUT_TIME_AND_POSITIONS
//#define RVO_SEED_RANDOM_NUMBER_GENERATOR

using System.Collections;
using System.Collections.Generic;
using System.IO;
using System;
using System.Text;
using UnityEngine;
using RVO;


using Vector2 = RVO.Vector2;
using Random = System.Random;


public class navigation : MonoBehaviour
{
    /* Store the goals of the agents. */
    IList<Vector2> goals;

    /** Random number generator. */
    Random random;

    void Start()
    {
        goals = new List<Vector2>();

#if RVO_SEED_RANDOM_NUMBER_GENERATOR
        random = new Random();
#else
        random = new System.Random(0);
#endif
        setupScenario();
    }

    void setupScenario()
    {
        /* Specify the global time step of the simulation. */
        Simulator.Instance.setTimeStep(0.25f);

        /*
            * Specify the default parameters for agents that are subsequently
            * added.
            */
        Simulator.Instance.setAgentDefaults(15.0f, 10, 5.0f, 5.0f, 2.0f, 2.0f, new Vector2(0.0f, 0.0f));

        /*
            * Add agents, specifying their start position, and store their
            * goals on the opposite side of the environment.
            */
        for (int i = 0; i < 5; ++i)
        {
            for (int j = 0; j < 5; ++j)
            {
                Simulator.Instance.addAgent(new Vector2(55.0f + i * 10.0f, 55.0f + j * 10.0f));
                goals.Add(new Vector2(-75.0f, -75.0f));

                Simulator.Instance.addAgent(new Vector2(-55.0f - i * 10.0f, 55.0f + j * 10.0f));
                goals.Add(new Vector2(75.0f, -75.0f));

                Simulator.Instance.addAgent(new Vector2(55.0f + i * 10.0f, -55.0f - j * 10.0f));
                goals.Add(new Vector2(-75.0f, 75.0f));

                Simulator.Instance.addAgent(new Vector2(-55.0f - i * 10.0f, -55.0f - j * 10.0f));
                goals.Add(new Vector2(75.0f, 75.0f));
            }
        }

        /*
            * Add (polygonal) obstacles, specifying their vertices in
            * counterclockwise order.
            */
        IList<Vector2> obstacle1 = new List<Vector2>();
        obstacle1.Add(new Vector2(-10.0f, 40.0f));
        obstacle1.Add(new Vector2(-40.0f, 40.0f));
        obstacle1.Add(new Vector2(-40.0f, 10.0f));
        obstacle1.Add(new Vector2(-10.0f, 10.0f));
        Simulator.Instance.addObstacle(obstacle1);

        IList<Vector2> obstacle2 = new List<Vector2>();
        obstacle2.Add(new Vector2(10.0f, 40.0f));
        obstacle2.Add(new Vector2(10.0f, 10.0f));
        obstacle2.Add(new Vector2(40.0f, 10.0f));
        obstacle2.Add(new Vector2(40.0f, 40.0f));
        Simulator.Instance.addObstacle(obstacle2);

        IList<Vector2> obstacle3 = new List<Vector2>();
        obstacle3.Add(new Vector2(10.0f, -40.0f));
        obstacle3.Add(new Vector2(40.0f, -40.0f));
        obstacle3.Add(new Vector2(40.0f, -10.0f));
        obstacle3.Add(new Vector2(10.0f, -10.0f));
        Simulator.Instance.addObstacle(obstacle3);

        IList<Vector2> obstacle4 = new List<Vector2>();
        obstacle4.Add(new Vector2(-10.0f, -40.0f));
        obstacle4.Add(new Vector2(-10.0f, -10.0f));
        obstacle4.Add(new Vector2(-40.0f, -10.0f));
        obstacle4.Add(new Vector2(-40.0f, -40.0f));
        Simulator.Instance.addObstacle(obstacle4);

        /*
            * Process the obstacles so that they are accounted for in the
            * simulation.
            */
        Simulator.Instance.processObstacles();
    }

#if RVO_OUTPUT_TIME_AND_POSITIONS
    void updateVisualization()
    {
        /* Output the current global time. */
        Debug.Log(Simulator.Instance.getGlobalTime());

        /* Output the current position of all the agents. */
        String context = "";
        for (int i = 0; i < Simulator.Instance.getNumAgents(); ++i)
        {
            //Console.Write(" {0}", Simulator.Instance.getAgentPosition(i));
            context = context + Simulator.Instance.getAgentPosition(i).ToString();
        }
        Debug.Log(context);
    }
#endif

    void setPreferredVelocities()
    {
        /*
            * Set the preferred velocity to be a vector of unit magnitude
            * (speed) in the direction of the goal.
            */
        for (int i = 0; i < Simulator.Instance.getNumAgents(); ++i)
        {
            Vector2 goalVector = goals[i] - Simulator.Instance.getAgentPosition(i);

            if (RVOMath.absSq(goalVector) > 1.0f)
            {
                goalVector = RVOMath.normalize(goalVector);
            }

            Simulator.Instance.setAgentPrefVelocity(i, goalVector);

            /* Perturb a little to avoid deadlocks due to perfect symmetry. */
            float angle = (float)random.NextDouble() * 2.0f * (float)Math.PI;
            float dist = (float)random.NextDouble() * 0.0001f;

            Simulator.Instance.setAgentPrefVelocity(i, Simulator.Instance.getAgentPrefVelocity(i) +
                dist * new Vector2((float)Math.Cos(angle), (float)Math.Sin(angle)));
        }
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

    private void Update()
    {
        Debug.Log("start");
#if RVO_OUTPUT_TIME_AND_POSITIONS
        updateVisualization();
#endif
        Debug.Log("after updateVisualization");
        setPreferredVelocities();
        Debug.Log("after setPreferredVelocities");
        Simulator.Instance.doStep();
        Debug.Log("after doStep");

        Debug.Log(reachedGoal());
    }
}
