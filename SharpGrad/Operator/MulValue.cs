﻿using SharpGrad.DifEngine;
using System;
using System.Collections.Generic;
using System.Linq.Expressions;
using System.Numerics;

namespace SharpGrad.Operators
{
    public class MulValue<TType> : BinaryOpValue<TType>
        where TType : INumber<TType>
    {
        public MulValue(Value<TType> left, Value<TType> right)
            : base("*", left, right)
        { }

        internal override Expression GetForwardComputation(Dictionary<Value<TType>, Expression> variableExpressions)
            => Expression.Multiply(LeftOperand.GetAsOperand(variableExpressions), RightOperand.GetAsOperand(variableExpressions));

        protected override void ComputeLeftGradient(Dictionary<Value<TType>, Expression> variableExpressions, Dictionary<Value<TType>, Expression> gradientExpressions, List<Expression> expressionList)
        {
            Expression grad = gradientExpressions[this];
            Expression right = variableExpressions[RightOperand];
            Expression gr = Expression.Multiply(grad, right);
            AssignGradientExpession(gradientExpressions, expressionList, LeftOperand, gr);
        }

        protected override void ComputeRightGradient(Dictionary<Value<TType>, Expression> variableExpressions, Dictionary<Value<TType>, Expression> gradientExpressions, List<Expression> expressionList)
        {
            Expression grad = gradientExpressions[this];
            Expression left = variableExpressions[LeftOperand];
            Expression lg = Expression.Multiply(left, grad);
            AssignGradientExpession(gradientExpressions, expressionList, RightOperand, lg);
        }
    }
}