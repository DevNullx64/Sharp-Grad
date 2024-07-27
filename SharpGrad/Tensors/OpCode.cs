using System;

namespace SharpGrad.Tensors
{
    /// <summary>
    /// An operation to perform using the KPU
    /// </summary>
    public enum OpCode : short
    {

        #region Flags
        /// <summary>
        /// If set, the operation is commutative. Otherwise, it is not.
        /// </summary>
        Commutative = 0x200,

        /// <summary>
        /// If set, the operation is unary. Otherwise, it is binary.
        /// </summary>
        Unary = 0x400,

        /// <summary>
        /// If set, the operation is a reduction. Otherwise, it is not.
        /// </summary>
        Reduction = 0x800,
        #endregion

        #region Utilities
        None = -1,

        [Obsolete("!!! Not implemented yet !!!")]
        Reset = 0,

        /// <summary>
        /// Store the result in the left operand
        /// </summary>
        Store = 1 | Unary,
        #endregion

        #region Arithmetic
        /// <summary>
        /// Add the left operand to the right operand
        /// </summary>
        Add = 5 | Commutative,

        /// <summary>
        /// Subtract the right operand from the left operand
        /// </summary>
        Sub = 6,

        /// <summary>
        /// Negate the left operand
        /// </summary>
        Neg = Sub | Unary,

        /// <summary>
        /// Multiply the left operand by the right operand
        /// </summary>
        Mul = 7 | Commutative,

        /// <summary>
        /// Divide the left operand by the right operand
        /// </summary>
        Div = 8,
        #endregion

        #region Trigonometry !!! Not implemented yet !!!
        [Obsolete("!!! Not implemented yet !!!")]
        Pow = 16,
        [Obsolete("!!! Not implemented yet !!!")]
        Log = 17 | Unary,
        [Obsolete("!!! Not implemented yet !!!")]
        Exp = 18 | Unary,

        [Obsolete("!!! Not implemented yet !!!")]
        Abs = 19 | Unary,
        [Obsolete("!!! Not implemented yet !!!")]
        Sqrt = 20 | Unary,
        [Obsolete("!!! Not implemented yet !!!")]
        Sin = 21 | Unary,
        [Obsolete("!!! Not implemented yet !!!")]
        Cos = 22 | Unary,
        [Obsolete("!!! Not implemented yet !!!")]
        Tan = 23 | Unary,
        #endregion

        #region Reductions !!! Not implemented yet !!!
        [Obsolete("!!! Not implemented yet !!!")]
        Sum = 32 | Reduction,
        [Obsolete("!!! Not implemented yet !!!")]
        Prod = 33 | Reduction,
        [Obsolete("!!! Not implemented yet !!!")]
        Min = 34 | Reduction,
        [Obsolete("!!! Not implemented yet !!!")]
        Max = 35 | Reduction,
        [Obsolete("!!! Not implemented yet !!!")]
        Mean = 36 | Reduction,
        [Obsolete("!!! Not implemented yet !!!")]
        Var = 37 | Reduction,
        [Obsolete("!!! Not implemented yet !!!")]
        Std = 38 | Reduction,
        #endregion
    }
}