using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class S1 : MonoBehaviour {

    public S2 other;

    void Update()
    {
        other.DoSomething();
    }
}
