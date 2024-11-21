using SharpGrad.DifEngine;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;

namespace SharpGrad.NN
{
    public static partial class Loss
    {
        public static Value<TType> MSE<TType>(List<Value<TType>> Y, List<Value<TType>> Y_hat)
            where TType : unmanaged, IBinaryFloatingPointIeee754<TType>
        {
            Value<TType> loss = Value<TType>.Zero;
            for (int i = 0; i < Y.Count; i++)
            {
                var l = Y[i] - Y_hat[i];
                var nl = loss + (l * l);
                loss = nl;
            }
            return loss;
        }
    }
}
