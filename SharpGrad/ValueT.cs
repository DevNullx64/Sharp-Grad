using SharpGrad.ExprLambda;
using SharpGrad.Operators;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Linq.Expressions;
using System.Numerics;
using System.Reflection;

namespace SharpGrad.DifEngine
{
    /// <summary>
    /// Base class for all typed values in the computation graph.
    /// </summary>
    /// <typeparam name="TType">The numeric type of the value.</typeparam>
    public abstract class Value<TType>: Value, IGraphNode<Value<TType>>
        where TType : INumber<TType>
    {
        protected static readonly PropertyInfo thisIndexerProperty = typeof(Value<TType>).GetProperty("Item", typeof(TType))!;

        public static readonly Expression ExpressionZero = Expression.Constant(TType.Zero);
        public static readonly Expression ExpressionOne = Expression.Constant(TType.One);

        private static int InstanceCount = 0;
        public static readonly Constant<TType> E = new(TType.CreateSaturating(Math.E), "e");
        public static readonly Constant<TType> Zero = new(TType.Zero, "0");
        public virtual void InitValueForForward() { }
        public Value(IReadOnlyList<Dimension> shape, string name, bool isParallelBarrier, params Value<TType>[] childs):
            base(name, typeof(TType), new Shape([.. shape.Where(e => e.Size > 1).Distinct()]), isParallelBarrier, childs)
        {
            int length = Size;
            data = new TType[length];
            gradient = new TType[length];
            InitValueForForward();
        }

        public new Value<TType>[] Operands => (Value<TType>[])base.Operands;
        protected TType[] data;
        public virtual TType[] Data => data;

        public TType this[Dimdices indices]
        {
            get
            {
                int i = Shape.GetLinearIndex(indices);
                return data[i];
            }
            internal set
            {
                int i = Shape.GetLinearIndex(indices);
                data[i] = value;
            }
        }

        internal TType this[int index, Shape shape]
        {
            get
            {
                int i = Shape.GetLinearIndex(index, shape);
                return data[i];
            }
            set
            {
                int i = Shape.GetLinearIndex(index, shape);
                data[i] = value;
            }
        }

        public Expression Get(Expression index)
            => Expression.MakeIndex(Expression.Constant(this), thisIndexerProperty, [index]);

        private TType[] gradient;
        public TType GetGradient(Dimdices indices)
        {
            int i = Shape.GetLinearIndex(indices);
            return gradient[i];
        }

        public void SetGradient(Dimdices indices, TType value)
        {
            int i = Shape.GetLinearIndex(indices);
            gradient[i] = value;
        }

        internal void InnerDFS(List<Value<TType>> topOSort, Dictionary<Value<TType>, int> usageCount)
        {
            if (usageCount.TryAdd(this, 0))
            {
                for (int i = 0; i < Operands.Length; i++)
                {
                    if (Operands[i] is ReduceOperation<TType> r)
                    {
                        if (usageCount.TryAdd(r, 0))
                        {
                            topOSort.Add(r);
                        }
                        usageCount[r]++;
                    }
                    else
                    {
                        Operands[i].InnerDFS(topOSort, usageCount);
                    }
                }
                topOSort.Add(this);
            }
            else
            {
                usageCount[this]++;
            }
        }

        private static int gradientCount = 0;
        internal void AssignGradientExpession(Dictionary<Value<TType>, Expression> gradientExpressions, List<Expression> expressionList, Expression index, Value<TType> LeftOperand, Expression gradientExpression)
        {
            if (!gradientExpressions.TryGetValue(LeftOperand, out Expression? leftGrad))
            {
                leftGrad = Expression.Variable(typeof(TType), $"grad{gradientCount++}");
                gradientExpressions[LeftOperand] = leftGrad;
                // Get the gradient of the left operand
                MethodInfo getGradientMethod = typeof(Value<TType>).GetMethod(nameof(GetGradient), [typeof(Dimdices)])!;
                Expression getGradientCall = Expression.Call(Expression.Constant(LeftOperand), getGradientMethod, index);
                // Assign the gradient to the left operand
                expressionList.Add(Expression.Assign(leftGrad, getGradientCall));
            }
            expressionList.Add(Expression.AddAssign(leftGrad, gradientExpression));


            // Add this grad to this value using a call to AddGradient
            MethodInfo addGradientMethod = typeof(Value<TType>).GetMethod(nameof(SetGradient), [typeof(Dimdices), typeof(TType)])!;
            Expression addGradientCall = Expression.Call(Expression.Constant(this), addGradientMethod, index, leftGrad);
            expressionList.Add(addGradientCall);
        }

        public abstract bool GetAsOperand(Dictionary<Value<TType>, Expression> variableExpressions, List<Expression> forwardExpressionList, Expression index, out Expression? operand);
        internal abstract Expression GetForwardComputation(Dictionary<Value<TType>, Expression> variableExpressions, List<Expression> forwardExpressionList, Expression index);
        public void BuildForward(Dictionary<Value<TType>, Expression> variableExpressions, List<Expression> forwardExpressionList, Expression index)
            => _ = GetAsOperand(variableExpressions, forwardExpressionList, index, out var _);
        public Expression GetAsOperand(Dictionary<Value<TType>, Expression> variableExpressions, Expression index)
        {
            List<Expression> forwardExpressionList = [];
            if (GetAsOperand(variableExpressions, forwardExpressionList, index, out var operand)
                && forwardExpressionList.Count == 0)
            {
                return operand!;
            }
            else
            {
                throw new InvalidOperationException($"Expression list should be empty. Found {forwardExpressionList.Count} expressions.");
            }
        }

        #region BASIC ARITHMETIC OPERATIONS
        public static AddValue<TType> Add(Value<TType> left, Value<TType> right) => new(left, right);
        public static AddValue<TType> operator +(Value<TType> left, Value<TType> right) => Add(left, right);

        public static NegValue<TType> Neg(Value<TType> operand) => new(operand);
        public static NegValue<TType> operator -(Value<TType> operand) => Neg(operand);

        public static SubValue<TType> Sub(Value<TType> left, Value<TType> right) => new(left, right);
        public static SubValue<TType> operator -(Value<TType> left, Value<TType> right) => Sub(left, right);

        public static MulValue<TType> Mul(Value<TType> left, Value<TType> right) => new(left, right);
        public static MulValue<TType> operator *(Value<TType> left, Value<TType> right) => Mul(left, right);

        public static DivValue<TType> Div(Value<TType> left, Value<TType> right) => new(left, right);
        public static DivValue<TType> operator /(Value<TType> left, Value<TType> right) => Div(left, right);
        #endregion

        protected void InitGradientForBackward()
        {
            Array.Fill(gradient, TType.One);
        }

        public void ResetGradient()
        {
            Array.Clear(gradient);
            if (Operands.Length > 0)
            {
                foreach (var child in Operands)
                {
                    child.ResetGradient();
                }
            }
        }

        public static implicit operator Value<TType>(TType d)
            => new Constant<TType>(d, $"v{InstanceCount++}");
    }
}