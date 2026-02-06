namespace SharpGrad.DifEngine.SyntaxBuilder.Operations
{
    /// <summary>
    /// Defines kinds of unary operations.
    /// </summary>
    public enum KindUnary
    {
        Negate = 1 | KindCategory.Unary | KindProperty.Differentiable,
        Reciprocal = 2 | KindCategory.Unary | KindProperty.Differentiable,
        Sqrt = 3 | KindCategory.Unary | KindProperty.Differentiable,
        Sq = Sqrt | KindProperty.Inverse,
        Exp = 4 | KindCategory.Unary | KindProperty.Differentiable,
        Log = Exp | KindProperty.Inverse | KindProperty.Differentiable,
        Log10 = 5 | KindCategory.Unary | KindProperty.Differentiable,
        Sin = 6 | KindCategory.Unary | KindProperty.Differentiable,
        Asin = Sin | KindProperty.Inverse | KindProperty.ConditionallyDifferentiable,
        Cos = 7 | KindCategory.Unary | KindProperty.Differentiable,
        Acos = Cos | KindProperty.Inverse | KindProperty.ConditionallyDifferentiable,
        Tan = 8 | KindCategory.Unary | KindProperty.Differentiable,
        Atan = Tan | KindProperty.Inverse | KindProperty.ConditionallyDifferentiable,
        Sinh = 9 | KindCategory.Unary | KindProperty.Differentiable,
        Asinh = Sinh | KindProperty.Inverse | KindProperty.ConditionallyDifferentiable,
        Cosh = 10 | KindCategory.Unary | KindProperty.Differentiable,
        Acosh = Cosh | KindProperty.Inverse | KindProperty.ConditionallyDifferentiable,
        Tanh = 11 | KindCategory.Unary | KindProperty.Differentiable,
        Atanh = Tanh | KindProperty.Inverse | KindProperty.ConditionallyDifferentiable,
        Floor = 12 | KindCategory.Unary,
        Ceil = 13 | KindCategory.Unary,
        Trunc = 14 | KindCategory.Unary,
        Round = 15 | KindCategory.Unary,
        Abs = 16 | KindCategory.Unary | KindProperty.PartiallyDifferentiable,
        Sign = 17 | KindCategory.Unary,
        Cast = 18 | KindCategory.Unary | KindProperty.Differentiable,
    }
}
