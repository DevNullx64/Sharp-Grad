namespace SharpGrad.DifEngine.SyntaxBuilder.Operations
{
    /// <summary>
    /// Defines kinds of graph nodes.
    /// </summary>
    public enum KindGraphNode
    {
        // Value operations
        Constant = KindValue.Constant,
        Variable = KindValue.Variable,

        // Function operations
        DifferentiableFunction = KindFunction.DifferentiableFunction,
        Function = KindFunction.Function,

        // Unary operations
        Negate = KindUnary.Negate,
        Reciprocal = KindUnary.Reciprocal,
        Sqrt = KindUnary.Sqrt,
        Sq = KindUnary.Sq,
        Exp = KindUnary.Exp,
        Log = KindUnary.Log,
        Log10 = KindUnary.Log10,
        Sin = KindUnary.Sin,
        Asin = KindUnary.Asin,
        Cos = KindUnary.Cos,
        Acos = KindUnary.Acos,
        Tan = KindUnary.Tan,
        Atan = KindUnary.Atan,
        Sinh = KindUnary.Sinh,
        Asinh = KindUnary.Asinh,
        Cosh = KindUnary.Cosh,
        Acosh = KindUnary.Acosh,
        Tanh = KindUnary.Tanh,
        Atanh = KindUnary.Atanh,
        Floor = KindUnary.Floor,
        Ceil = KindUnary.Ceil,
        Trunc = KindUnary.Trunc,
        Round = KindUnary.Round,
        Abs = KindUnary.Abs,
        Sign = KindUnary.Sign,
        Cast = KindUnary.Cast,

        // Binary operations
        Add = KindBinary.Add,
        Subtract = KindBinary.Subtract,
        Multiply = KindBinary.Multiply,
        Divide = KindBinary.Divide,
        Power = KindBinary.Power,
        Modulo = KindBinary.Modulo,
        Min = KindBinary.Min,
        Max = KindBinary.Max,

        // Reduction operations
        Sum = KindReduction.Sum,
        Product = KindReduction.Product,
        // MaxOf = KindReduction.MaxOf,
        // MinOf = KindReduction.MinOf,

        // Comparison operations
        Equal = KindComparison.Equal,
        NotEqual = KindComparison.NotEqual,
        GreaterThan = KindComparison.GreaterThan,
        GreaterEqual = KindComparison.GreaterEqual,
        LessThan = KindComparison.LessThan,
        LessEqual = KindComparison.LessEqual,
    }
}
