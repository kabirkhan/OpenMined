using System;
using System.Collections.Generic;
using OpenMined.Network.Controllers;
using OpenMined.Network.Utils;
using UnityEngine;


namespace OpenMined.Syft.Optim
{
    public class SGD
    {
        private SyftController ctrl;
        private List<int> parameters;
        private float lr;
        private float momentum;
        private float decay;
        
        // Should we put a check incase this variable overflows?
        protected static volatile int nCreated = 0;
        protected int id;
        
        public int Id
        {
            get { return id; }
            protected set { id = value; }
        }

        public SGD(SyftController ctrl_, List<int> parameters_, float lr_, float momentum_, float decay_)
        {
            this.ctrl = ctrl_;
            this.parameters = parameters_;
            this.lr = lr_;
            this.momentum = momentum_;
            this.decay = decay_;
            
            #pragma warning disable 420
            id = System.Threading.Interlocked.Increment(ref nCreated);
            ctrl.addOptimizer(this);
        }

        public void ZeroGrad()
        {
            foreach (int param_index in parameters)
                if(ctrl.floatTensorFactory.Get(param_index).Grad != null)
                    ctrl.floatTensorFactory.Get(param_index).Grad.Zero_();
        }

        public void Step(int batch_size, int iteration)
        {            
            foreach (int param_index in parameters)
            {
                var param = ctrl.floatTensorFactory.Get(param_index);
                var vel = param.createZerosTensorLike();
                vel = vel.Mul(momentum).Add(param.Grad.Mul(1.0F - momentum));
                param.Sub(vel.Mul(lr/(float)batch_size), inline:true);
            }

            if (this.decay > 0)
            {
                this.lr *= 1.0F / (1.0F + this.decay * iteration);
            }
        }
        
        public string ProcessMessage (Command msgObj, SyftController ctrl)
        {

            switch (msgObj.functionCall)
            {
                case "zero_grad":
                    ZeroGrad();
                    return "";
                case "step":
                    Step(int.Parse(msgObj.tensorIndexParams[0]), int.Parse(msgObj.tensorIndexParams[1]));
                    return "";
               
            }

            throw new InvalidOperationException("Could not find function for command:" + msgObj.functionCall);

        }
    }
}