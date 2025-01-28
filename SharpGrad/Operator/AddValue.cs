﻿using SharpGrad.DifEngine;
using System;
using System.Collections.Generic;
using System.Linq.Expressions;
using System.Numerics;

namespace SharpGrad.Operators
{
    public class AddValue<TType> : BinaryOpValue<TType>
        where TType : INumber<TType>
    {
        public AddValue(Value<TType> left, Value<TType> right)
            : base("+", left, right)
        {
        }


        protected override Expression GetForwardComputation(Dictionary<Value<TType>, Expression> variableExpressions)
            => Expression.Add(LeftOperand.GetAsOperand(variableExpressions), RightOperand.GetAsOperand(variableExpressions));


        protected override void Backward()
        {
            LeftOperand.Grad += Grad;
            RightOperand.Grad += Grad;
        }

    }
}