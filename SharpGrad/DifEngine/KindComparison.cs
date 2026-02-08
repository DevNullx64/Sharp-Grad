namespace SharpGrad.DifEngine.SyntaxBuilder
{
    /// <summary>
    /// Defines kinds of comparison operations.
    /// </summary>
    public enum KindComparison
    {
        Equal = 1 | KindCategory.Comparison,
        NotEqual = Equal | KindProperty.Inverse,
        GreaterThan = 2 | KindCategory.Comparison,
        GreaterEqual = GreaterThan | KindProperty.Inverse,
        LessThan = 3 | KindCategory.Comparison,
        LessEqual = LessThan | KindProperty.Inverse,
    }
}
