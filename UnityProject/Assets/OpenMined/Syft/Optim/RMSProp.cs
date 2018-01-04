using System;
using System.Collections.Generic;
using OpenMined.Network.Controllers;
using OpenMined.Network.Utils;
using UnityEngine;


namespace OpenMined.Syft.Optim
{
    public class RMSProp: Optimizer
    {
        private float rho;
        private float epsilon;
        private List<int> squares;

        public RMSProp(SyftController ctrl_, List<int> parameters_, float lr_, float rho_, float epsilon_, float decay_)
        {
            this.ctrl = ctrl_;
            this.parameters = parameters_;
            this.lr = lr_;
            this.rho = rho_;
            this.epsilon = epsilon_;
            this.decay = decay_;

            #pragma warning disable 420
            id = System.Threading.Interlocked.Increment(ref nCreated);
            ctrl.addOptimizer(this);

            Init(parameters);
        }

        public void Init(List<int> parameters)
        {
            if (this.squares != null)
            {
                return;
            }
            this.squares = new List<int>();

            foreach (int param_index in parameters)
            {
                var param = ctrl.floatTensorFactory.Get(param_index);
                var sInit = param.createZerosTensorLike();
                this.squares.Add(sInit.Id);
            }
        }

        override public void Step(int batch_size, int iteration)
        {
            // Debug.LogFormat("<color=green>RMSProp Step: lr: {0} rho: {1} ep: {2} decay: {3}</color>", lr, rho, epsilon, decay);
            if (epsilon == 0) {
                epsilon = 0.00000001f;
            }

            for (int i = 0; i < parameters.Count; i++)
            {
                var param = ctrl.floatTensorFactory.Get(parameters[i]);
                var square = ctrl.floatTensorFactory.Get(squares[i]);

                // Debug.LogFormat("<color=green>BEFORE RMSProp Step: \n param: {0} \n square: {1}</color>", param.Print(), square.Print());
                square.Mul(rho, inline: true);
                
                // Debug.LogFormat("<color=red>GRAD RMSProp Step: \n {0}</color>", param.Grad.Print());

                var gradSqr = param.Grad.Pow(2.0F);
                // Debug.LogFormat("<color=orange>GRAD SQUARE RMSProp Step: \n {0}</color>", gradSqr.Print());
                var betaGradSqr = gradSqr.Mul(1.0F - rho);
                // Debug.LogFormat("<color=magenta>BETA GRAD SQUARE RMSProp Step: \n {0}</color>", betaGradSqr.Print());

                square.Add(betaGradSqr, inline: true);
                // Debug.LogFormat("<color=cyan>SQUARE: \n {0}</color>", square.Print());

                var div = square.Div(square.Sqrt().Add(epsilon));
                // Debug.LogFormat("<color=blue>DIV: \n {0}</color>", div.Print());

                param.Sub(div.Mul(lr/(float)batch_size), inline:true);
                // Debug.LogFormat("<color=green>AFTER RMSProp Step: \n param: {0} \n square: {1}</color>", param.Print(), square.Print());
            }

            if (this.decay > 0)
            {
                this.lr *= 1.0F / (1.0F + this.decay * iteration);
            }
        }
        
        override public string ProcessMessage (Command msgObj, SyftController ctrl)
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
