namespace SharpGrad.Tensors.Operators
{
    internal class CastDoubleToFloat : IExecutor1<double, float>
    {
        public static Shape ResultingShape(Shape operand1) => operand1;
        public static float Exec(double operand1) => (float)operand1;
        public static double Backward(double operand1, float grad) => grad;
    }

}