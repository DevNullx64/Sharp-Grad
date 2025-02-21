﻿using SharpGrad.DifEngine;
using SharpGrad.Operator;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Numerics;

namespace SharpGrad.NN
{
    public static partial class Loss
    {
        public static NariOpValue<TType> MSE<TType>(Value<TType>[] Y, Value<TType>[] Y_hat)
            where TType : INumber<TType>
        {
            NariOpValue<TType> loss = Y[0] - Y_hat[0];
            loss *= loss;
            for (int i = 1; i < Y.Length; i++)
            {
                var nl = Y[i] - Y_hat[i];
                loss += nl * nl;
            }
            return loss;
        }
    }
}
