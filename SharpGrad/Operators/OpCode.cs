namespace SharpGrad.Operators
{
    public enum OpCode : short
    {
        /// <summary>
        /// Binary mask for <see cref="OpCode"/> to get the raw code.
        /// </summary>
        CodeMask = 0x0F,

        None = 0,
        Reset = 1,
        Store = 2,

        /// <summary>
        /// IsFunction is set for <see cref="OpCode"/> that are functions.
        /// </summary>
        IsUnary = 0x10,
        #region Functions
        Neg = SharedFuncCode.Neg | IsUnary,
        Log = SharedFuncCode.Log | IsUnary,
        Exp = SharedFuncCode.Exp | IsUnary,
        Sqrt = SharedFuncCode.Sqrt | IsUnary,
        Abs = SharedFuncCode.Abs | IsUnary,
        Sin = SharedFuncCode.Sin | IsUnary,
        Cos = SharedFuncCode.Cos | IsUnary,
        Tan = SharedFuncCode.Tan | IsUnary,
        Tanh = SharedFuncCode.Tanh | IsUnary,
        #endregion

        /// <summary>
        /// IsOperator is set for <see cref="OpCode"/> that are operators.
        /// </summary>
        IsBinary = 0x20,
        #region Operators
        IsCommutative = SharedOpCode.IsCommutative | IsBinary,
        Add = SharedOpCode.Add | IsBinary,
        Sub = SharedOpCode.Sub | IsBinary,
        Mul = SharedOpCode.Mul | IsBinary,
        Div = SharedOpCode.Div | IsBinary,
        Pow = SharedOpCode.Pow | IsBinary,
        #endregion

        /// <summary>
        /// IcReduction is set for <see cref="OpCode"/> that are reductions.
        /// </summary>
        IsReduction = 0x30,
        #region Reductions
        Min = SharedReduceCode.Min | IsReduction,
        Max = SharedReduceCode.Max | IsReduction,
        Var = SharedReduceCode.Var | IsReduction,
        Std = SharedReduceCode.Std | IsReduction,
        Sum = SharedReduceCode.Sum | IsReduction,
        Prod = SharedReduceCode.Prod | IsReduction,
        Mean = SharedReduceCode.Mean | IsReduction,
        #endregion
    }
}