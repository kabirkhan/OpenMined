﻿using System;
using System.Collections.Generic;
using OpenMined.Network.Controllers;
using OpenMined.Network.Utils;
using UnityEngine;


namespace OpenMined.Syft.Optim
{
    public class SGD: Optimizer
    {
        private float momentum;
        private List<int> velocities;

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

            // Create ZeroLike copies of parameters FloatTensors to store 
            // momentum velocity updates. 
            // (If momentum == 0 the velocity tensors will always be Zero)
            Init(parameters);
        }

        public void Init(List<int> parameters)
        {
            if (this.velocities != null)
            {
                return;
            }
            this.velocities = new List<int>();

            foreach (int param_index in parameters)
            {
                var param = ctrl.floatTensorFactory.Get(param_index);
                var velInit = param.createZerosTensorLike();
                this.velocities.Add(velInit.Id);
            }
            // Debug.LogFormat("<color=green>INIT SGD: num_params: {0} num_vel: {1}</color>", parameters.Count, velocities.Count);
        }

        public override void Step(int batch_size, int iteration)
        {            
            for (int i = 0; i < parameters.Count; i++)
            {
                var param = ctrl.floatTensorFactory.Get(parameters[i]);
                Debug.LogFormat("<color=red>GRAD RMSProp Step: \n {0}</color>", param.Grad.Print());
                var vel = ctrl.floatTensorFactory.Get(velocities[i]);
                
                vel.Mul(momentum, inline: true).Add(param.Grad.Mul(1.0F - momentum), inline: true);
                param.Sub(vel.Mul(lr/(float)batch_size), inline: true);
            }

            // Adjust learning rate with decay
            if (this.decay > 0)
            {
                this.lr *= 1.0F / (1.0F + this.decay * iteration);
            }
        }
        
        public override string ProcessMessage (Command msgObj, SyftController ctrl)
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