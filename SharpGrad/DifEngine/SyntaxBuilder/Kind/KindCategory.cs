namespace SharpGrad.DifEngine.SyntaxBuilder.Operations
{
    /// <summary>
    /// Defines categories of graph node kinds.
    /// </summary>
    public enum KindCategory : int
    {
        Mask = unchecked((int)0b1110_0000_0000_0000_0000_0000_0000_0000),
        Value = 0 << 29,
        Unary = 1 << 29,
        Binary = 2 << 29,
        Reduction = 3 << 29,
        Comparison = 4 << 29,
    }
}
