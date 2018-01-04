using System;
using System.Collections.Generic;
using OpenMined.Network.Controllers;
using OpenMined.Network.Utils;
using UnityEngine;


namespace OpenMined.Syft.Optim
{
    public class Adam: Optimizer
    {
        private float beta1;
        private float beta2;
        private float epsilon;
        private int t;
        private List<int> velocities;
        private List<int> squares;

        public Adam(SyftController ctrl_, List<int> parameters_, float lr_, float beta1_, float beta2_, float epsilon_, float decay_)
        {
            this.ctrl = ctrl_;
            this.parameters = parameters_;
            this.lr = lr_;
            this.beta1 = beta1_;
            this.beta2 = beta2_;
            this.epsilon = epsilon_;
            this.decay = decay_;
            this.t = 0;
            
            #pragma warning disable 420
            id = System.Threading.Interlocked.Increment(ref nCreated);
            ctrl.addOptimizer(this);
            Init(parameters);
        }

        public void Init(List<int> parameters)
        {
            if (velocities != null || squares != null)
            {
                return;
            }
            velocities = new List<int>();
            squares = new List<int>();

            foreach (int param_index in parameters)
            {
                var param = ctrl.floatTensorFactory.Get(param_index);
                
                var velInit = param.createZerosTensorLike();
                this.velocities.Add(velInit.Id);

                var sInit = param.createZerosTensorLike();
                this.squares.Add(sInit.Id);
            }
            Debug.LogFormat("<color=green>INIT SGD: num_params: {0} num_vel: {1}</color>", squares.Count, velocities.Count);
            Debug.LogFormat("<color=green>INIT SGD: {0} {1} {2} {3}</color>", lr, beta1, beta2, epsilon);
        }

        public override void Step(int batch_size, int iteration)
        {            
            for (int i = 0; i < parameters.Count; i++)
            {
                var param = ctrl.floatTensorFactory.Get(parameters[i]);
                var v = ctrl.floatTensorFactory.Get(velocities[i]);
                var s = ctrl.floatTensorFactory.Get(squares[i]);
                
                v.Mul(beta1, inline: true).Add(param.Grad.Mul(1.0F - beta1), inline: true);
                var vCorrected = v.Div(1.0F - (float)Math.Pow(beta1, t));

                s.Mul(beta2, inline: true).Add(param.Grad.Pow(2.0F).Mul(1.0F - beta2), inline: true);
                var sCorrected = s.Div(1.0F - (float)Math.Pow(beta2, t));

                var div = vCorrected.Mul(sCorrected.Sqrt().Add(epsilon));

                param.Sub(div.Mul(lr/(float)batch_size), inline: true);
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
