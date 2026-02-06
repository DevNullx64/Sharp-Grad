namespace SharpGrad.DifEngine.SyntaxBuilder.Operations
{
    /// <summary>
    /// Defines kinds of reduction operations.
    /// </summary>
    public enum KindReduction
    {
        Sum = KindBinary.Add | KindCategory.Reduction,
        Difference = KindBinary.Subtract | KindCategory.Reduction,
        Product = KindBinary.Multiply | KindCategory.Reduction,
        Quotient = KindBinary.Divide | KindCategory.Reduction,
        // MaxOf = KindBinary.Max | KindCategory.Reduction | KindProperty.Commutative | KindProperty.PartiallyDifferentiable,
        // MinOf = KindBinary.Min | KindCategory.Reduction | KindProperty.Commutative | KindProperty.PartiallyDifferentiable,
    }
}
