using System;
using System.Numerics;

namespace SharpGrad.Tensors.Operators
{
    public class OpBase2<T>
        where T : unmanaged, INumber<T>
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

}