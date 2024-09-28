using System;
using System.Numerics;

namespace SharpGrad.Tensors.Operators
{
    public class BaseOperation<T>
        where T : unmanaged, INumber<T>
    {
        /// <summary>
        /// Broadcasts the <see cref="Shape"/>s of the operands.
        /// </summary>
        /// <param name="left">The <see cref="Shape"/> of the first operand. </param>
        /// <param name="right">The <see cref="Shape"/> of the second operand. </param>
        /// <returns>The broadcasted <see cref="Shape"/>. </returns>
        /// <exception cref="InvalidOperationException"></exception>
        public static Shape ResultingShape(Shape left, Shape right)
        {
            if(left.IsScalar)
                return right;
            if(right is T)
                return left;
            if(left.Length != right.Length)
                throw new InvalidOperationException($"Cannot broadcast shapes {left} and {right}");
            return left;
        }
    }

}