namespace SharpGrad.DifEngine.SyntaxBuilder.Operations
{
    /// <summary>
    /// Defines kinds of binary operations.
    /// </summary>
    public enum KindBinary
    {
        Add = 1 | KindCategory.Binary | KindProperty.Commutative | KindProperty.Associative | KindProperty.Differentiable, 
        Subtract = Add | KindProperty.Inverse,
        Multiply = 2 | KindCategory.Binary | KindProperty.Commutative | KindProperty.Associative | KindProperty.Differentiable,
        Divide = Multiply | KindProperty.Inverse,
        Power = 3 | KindCategory.Binary | KindProperty.Differentiable,
        Modulo = 4 | KindCategory.Binary | KindProperty.Differentiable,
        Min = 5 | KindCategory.Binary | KindProperty.Commutative | KindProperty.PartiallyDifferentiable,
        Max = 6 | KindCategory.Binary | KindProperty.Commutative | KindProperty.PartiallyDifferentiable,
    }
}
