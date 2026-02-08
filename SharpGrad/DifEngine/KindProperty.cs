namespace SharpGrad.DifEngine.SyntaxBuilder
{
    /// <summary>
    /// Defines properties of graph node kinds.
    /// </summary>
    public enum KindProperty : int
    {
        Mask = 0b0001_1111_0000_0000_0000_0000_0000_0000,
        DifferentiabilityTypeMask = 0b0000_0011_0000_0000_0000_0000_0000_0000,
        NotDifferentiable = 0 << 24,
        Differentiable = 1 << 24,
        PartiallyDifferentiable = 2 << 24,
        ConditionallyDifferentiable = 3 << 24,
        Associative = 1 << 26,
        Commutative = 1 << 27,
        Inverse = 1 << 28,
    }
}
