using System;
using System.Collections.Generic;
using OpenMined.Network.Controllers;
using OpenMined.Network.Utils;
using UnityEngine;


namespace OpenMined.Syft.Optim
{
    public abstract class Optimizer
    {
        protected SyftController ctrl;
        protected List<int> parameters;
        protected float lr;
        protected float decay;
        
        // Should we put a check incase this variable overflows?
        protected static volatile int nCreated = 0;
        protected int id;
        
        public int Id
        {
            get { return id; }
            protected set { id = value; }
        }

        public void ZeroGrad()
        {
            foreach (int param_index in parameters)
                if(ctrl.floatTensorFactory.Get(param_index).Grad != null)
                    ctrl.floatTensorFactory.Get(param_index).Grad.Zero_();
        }

        public abstract void Step(int batch_size, int iteration);
        
        public abstract string ProcessMessage(Command msgObj, SyftController ctrl);
    }
}