using SharpGrad.DifEngine.SyntaxBuilder;
using System;
using System.Numerics;

namespace SharpGrad.Activation
{
    public static class ActivationExtensions
    {
        /// <summary>
        /// SiLU (Sigmoid Linear Unit) or Swish activation.
        /// SiLU(x) = x * ?(x) = x / (1 + exp(-x))
        /// </summary>
        /// <remarks>
        /// SiLU is smooth and differentiable everywhere, unlike ReLU.
        /// Often performs better than ReLU in deep networks.
        /// </remarks>
        public static Value<T> SiLU<T>(this Value<T> input)
            where T : struct, INumber<T>, IExponentialFunctions<T>
        {
            // SiLU(x) = x * sigmoid(x)
            var sigmoid = input.Sigmoid();
            return input * sigmoid;
        }

        /// <summary>
        /// Sigmoid activation: ?(x) = 1 / (1 + exp(-x))
        /// </summary>
        public static Value<T> Sigmoid<T>(this Value<T> input)
            where T : struct, INumber<T>, IExponentialFunctions<T>
        {
            // ?(x) = 1 / (1 + exp(-x))
            var negInput = -input;
            var expNeg = new UnaryComputedValue<T>(
                KindUnary.Exp, 
                negInput);
            var onePlusExp = Constant<T>.One + expNeg;
            return Constant<T>.One / onePlusExp;
        }

        /// <summary>
        /// Hyperbolic tangent activation.
        /// </summary>
        public static UnaryComputedValue<T> Tanh<T>(this Value<T> input)
            where T : struct, INumber<T>, IHyperbolicFunctions<T>
        {
            return new UnaryComputedValue<T>(
                KindUnary.Tanh, 
                input);
        }

        /// <summary>
        /// GELU (Gaussian Error Linear Unit) approximation.
        /// GELU(x) ? 0.5 * x * (1 + tanh(?(2/?) * (x + 0.044715 * x³)))
        /// </summary>
        public static Value<T> GELU<T>(this Value<T> input)
            where T : struct, INumber<T>, IHyperbolicFunctions<T>, IPowerFunctions<T>
        {
            // Constants
            var half = new Constant<T>(T.CreateTruncating(0.5), "0.5");
            var one = Constant<T>.One;
            var sqrtTwoOverPi = new Constant<T>(T.CreateTruncating(Math.Sqrt(2 / Math.PI)), "?(2/?)");
            var coeff = new Constant<T>(T.CreateTruncating(0.044715), "0.044715");

            // x³
            var xCubed = input * input * input; // or input.Pow(three)

            // 0.044715 * x³
            var term = coeff * xCubed;
            
            // x + 0.044715 * x³
            var inner = input + term;
            
            // ?(2/?) * (x + 0.044715 * x³)
            var scaled = sqrtTwoOverPi * inner;
            
            // tanh(?(2/?) * (x + 0.044715 * x³))
            var tanhScaled = scaled.Tanh();
            
            // 1 + tanh(...)
            var onePlusTanh = one + tanhScaled;
            
            // 0.5 * x * (1 + tanh(...))
            return half * input * onePlusTanh;
        }

        /// <summary>
        /// ReLU (Rectified Linear Unit) using Max operation.
        /// ReLU(x) = max(0, x)
        /// </summary>
        /// <remarks>
        /// Note: Uses subgradient convention at x=0 (gradient flows to left operand).
        /// </remarks>
        public static BinaryComputedValue<T> ReLU<T>(this Value<T> input)
            where T : struct, INumber<T>
        {
            return new BinaryComputedValue<T>(
                KindBinary.Max,
                Constant<T>.Zero,
                input);
        }

        /// <summary>
        /// Leaky ReLU activation.
        /// LeakyReLU(x) = max(alpha * x, x)
        /// </summary>
        public static BinaryComputedValue<T> LeakyReLU<T>(this Value<T> input, T alpha)
            where T : struct, INumber<T>
        {
            var alphaConst = new Constant<T>(alpha, $"?({alpha})");
            var scaled = alphaConst * input;
            return new BinaryComputedValue<T>(
                KindBinary.Max,
                scaled,
                input);
        }
    }
}
