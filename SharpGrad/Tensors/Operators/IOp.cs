using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;

namespace SharpGrad.Tensors.Operators
{
    public interface  IAggregator<TOperand1, TResult>
    {
        abstract static Shape ResultingShape(Shape operand1);
        abstract static TResult Exec(TOperand1[] operand1);
        abstract static TOperand1[] Backward(TOperand1[] operand1, TResult grad);
    }

    internal class Aggregator<T> where T : unmanaged, INumber<T>
    {
        public static Shape ResultingShape(Shape operand1) => operand1[..^1];
    }

    internal class SumOp<T> : Aggregator<T>, IAggregator<T, T> where T : unmanaged, INumber<T>
    {
        public static T Exec(T[] operand1)
        {
            T sum = operand1[0];
            for (int i = 1; i < operand1.Length; i++)
                sum += operand1[i];
            return sum;
        }

        public static T[] Backward(T[] operand1, T grad)
        {
            T[] grads = new T[operand1.Length];
            for (int i = 0; i < operand1.Length; i++)
                grads[i] = grad;
            return grads;
        }
    }


    public interface IExecutor1<TOperand1, TResult>
    {
        abstract static Shape ResultingShape(Shape operand1);
        abstract static TResult Exec(TOperand1 operand1);
        abstract static TOperand1 Backward(TOperand1 operand1, TResult grad);
    }
    internal class OpBase1<T> where T : unmanaged, INumber<T>
    {
        public static Shape ResultingShape(Shape operand1) => operand1;
    }

    internal class NegOp<T> : OpBase1<T>, IExecutor1<T, T> where T : unmanaged, INumber<T>
    {
        public static T Backward(T operand1, T grad) => -grad;
        public static T Exec(T operand1) => -operand1;
    }
    internal class LogOp<T> : OpBase1<T>, IExecutor1<T, T> where T : unmanaged, INumber<T>, ILogarithmicFunctions<T>
    {
        public static T Backward(T operand1, T grad) => grad / operand1;
        public static T Exec(T operand1) => T.Log(operand1);
    }


    public interface IExecutor2<TOperand1, TOperand2, TResult>
    {
        abstract static Shape ResultingShape(Shape operand1, Shape operand2);
        abstract static TResult Exec(TOperand1 operand1, TOperand2 operand2);
        abstract static (TOperand1, TOperand2) Backward(TOperand1 operand1, TOperand2 operand2, TResult grad);
    }
    internal class OpBase2<T> where T : unmanaged, INumber<T>
    {
        /// <summary>
        /// Broadcasts the <see cref="Shape"/>s of the operands.
        /// </summary>
        /// <param name="operand1">The <see cref="Shape"/> of the first operand. </param>
        /// <param name="operand2">The <see cref="Shape"/> of the second operand. </param>
        /// <returns>The broadcasted <see cref="Shape"/>. </returns>
        /// <exception cref="InvalidOperationException"></exception>
        public static Shape ResultingShape(Shape operand1, Shape operand2)
        {
            if(operand1.IsScalar)
                return operand2;
            if(operand2 is T)
                return operand1;
            if(operand1.Length != operand2.Length)
                throw new InvalidOperationException($"Cannot broadcast shapes {operand1} and {operand2}");
            return operand1;
        }
    }

    internal class CastDoubleToFloat : IExecutor1<double, float>
    {
        public static Shape ResultingShape(Shape operand1) => operand1;
        public static float Exec(double operand1) => (float)operand1;
        public static double Backward(double operand1, float grad) => grad;
    }
    internal class CastFloatToDouble : IExecutor1<float, double>
    {
        public static Shape ResultingShape(Shape operand1) => operand1;
        public static double Exec(float operand1) => operand1;
        public static float Backward(float operand1, double grad) => (float)grad;
    }

    internal class AddOp<T> : OpBase2<T>,  IExecutor2<T, T, T> where T : unmanaged, INumber<T>
    {
        public static (T, T) Backward(T operand1, T operand2, T grad) => (grad, grad);
        public static T Exec(T operand1, T operand2) => operand1 + operand2;
    }
    internal class SubOp<T> : OpBase2<T>, IExecutor2<T, T, T> where T : unmanaged, INumber<T>
    {
        public static (T, T) Backward(T operand1, T operand2, T grad) => (grad, -grad);
        public static T Exec(T operand1, T operand2) => operand1 - operand2;
    }
    internal class MulOp<T> : OpBase2<T>, IExecutor2<T, T, T> where T : unmanaged, INumber<T>
    {
        public static (T, T) Backward(T operand1, T operand2, T grad) => (operand2 * grad, operand1 * grad);
        public static T Exec(T operand1, T operand2) => operand1 * operand2;
    }
    internal class DivOp<T> : OpBase2<T>, IExecutor2<T, T, T> where T : unmanaged, INumber<T>
    {
        public static (T, T) Backward(T operand1, T operand2, T grad) => (grad / operand2, -grad * operand1 / (operand2 * operand2));
        public static T Exec(T operand1, T operand2) => operand1 / operand2;
    }

    internal class PowOp<T> : OpBase2<T>, IExecutor2<T, T, T> where T : unmanaged, INumber<T>, IPowerFunctions<T>, ILogarithmicFunctions<T>
    {
        public static (T, T) Backward(T operand1, T operand2, T grad) => (grad * operand2 * T.Pow(operand1, operand2 - T.One), grad * T.Log(operand1) * T.Pow(operand1, operand2));
        public static T Exec(T operand1, T operand2) => T.Pow(operand1, operand2);
    }

    internal class ExpOp<T> : OpBase1<T>, IExecutor1<T, T> where T : unmanaged, INumber<T>, IExponentialFunctions<T>
    {
        public static T Backward(T operand1, T grad) => grad * T.Exp(operand1);
        public static T Exec(T operand1) => T.Exp(operand1);
    }

}