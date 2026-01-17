using System.Numerics;

namespace SharpGrad.Operators
{
    /// <summary>
    /// Represents a binary operation interface for tensors.
    /// </summary>
    /// <typeparam name="T">The numeric type of the tensor elements.</typeparam>
    public interface IOp2<T>
        where T : INumber<T>
    {
        /// <summary>
        /// Performs the binary operation on two tensor elements.
        /// </summary>
        /// <param name="left">The left operand.</param>
        /// <param name="right">The right operand.</param>
        /// <returns>The result of the operation.</returns>
        T Operate(T left, T right);

        /// <summary>
        /// Computes the gradients for the binary operation.
        /// </summary>
        /// <typeparam name="G">The gradient type.</typeparam>
        /// <param name="left">The left operand.</param>
        /// <param name="right">The right operand.</param>
        /// <param name="result">The result of the operation.</param>
        /// <param name="resultGradient">The gradient of the result.</param>
        /// <returns>A tuple containing the gradients for the left and right operands.</returns>
        (G Left, G Right) Backward<G>(T left, T right, T result, G resultGradient);
    }
}