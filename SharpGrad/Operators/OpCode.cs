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
        IsFunction = 0x10,
        #region Functions
        Neg = SharedFuncCode.Neg | IsFunction,
        Log = SharedFuncCode.Log | IsFunction,
        Exp = SharedFuncCode.Exp | IsFunction,
        Sqrt = SharedFuncCode.Sqrt | IsFunction,
        Abs = SharedFuncCode.Abs | IsFunction,
        Sin = SharedFuncCode.Sin | IsFunction,
        Cos = SharedFuncCode.Cos | IsFunction,
        Tan = SharedFuncCode.Tan | IsFunction,
        Tanh = SharedFuncCode.Tanh | IsFunction,
        #endregion

        /// <summary>
        /// IsOperator is set for <see cref="OpCode"/> that are operators.
        /// </summary>
        IsOperator = 0x20,
        #region Operators
        IsCommutative = SharedOpCode.IsCommutative | IsOperator,
        Add = SharedOpCode.Add | IsOperator,
        Sub = SharedOpCode.Sub | IsOperator,
        Mul = SharedOpCode.Mul | IsOperator,
        Div = SharedOpCode.Div | IsOperator,
        Pow = SharedOpCode.Pow | IsOperator,
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