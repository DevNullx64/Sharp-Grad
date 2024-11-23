using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SharpGrad.NN
{
    public static class DataSet
    {

        public class Data(List<int> X, List<int> Y)
        {
            public List<int> X = X;
            public List<int> Y = Y;
        }

        public static List<Data> GetDataSet(int n)
        {
            var rand = new Random();
            List<Data> dataset = [];
            for (int i = 0; i < n; i++)
            {
                List<int> X = [];
                List<int> Y = [];
                X.Add(rand.Next(-15, 15));
                X.Add(rand.Next(-15, 15));
                int x = X[0];
                int y = X[1];
                if (y > 2 * x + 5)
                    Y.Add(1);
                else
                    Y.Add(2);

                dataset.Add(new(X, Y));
            }
            return dataset;
        }

        private static readonly string UpperRow = "╔" + new string('═', Program.n) + "╗";
        private static readonly string LowerRow = "╚" + new string('═', Program.n) + "╝";

        public static int[,] GetMat(List<Data> v)
        {
            int[,] mat = new int[Program.n, Program.n];
            for (int i = 0; i < v.Count; i++)
                mat[v[i].X[0] + Program.n / 2, v[i].X[1] + Program.n / 2] = v[i].Y[0];
            return mat;
        }

        public static void Scatter(List<Data> x, List<Data> y)
        {
            int[,] matX = GetMat(x);
            int[,] matY = GetMat(y);

            Console.Write(UpperRow);
            Console.WriteLine(UpperRow);

            for (int r = 0; r < Program.n; r++)
            {
                Console.Write("║");
                for (int c = 0; c < Program.n; c++)
                {
                    switch (matX[r, c])
                    {
                        case 0:
                            Console.Write(" ");
                            break;
                        case 1:
                            Console.ForegroundColor = ConsoleColor.Red;
                            Console.Write("o");
                            break;
                        case 2:
                            Console.ForegroundColor = ConsoleColor.Blue;
                            Console.Write("o");
                            break;
                    }
                    Console.ResetColor();
                }
                Console.Write("║║");
                for (int j = 0; j < Program.n; j++)
                {
                    switch (matY[r, j])
                    {
                        case 0:
                            Console.Write(" ");
                            break;
                        case 1:
                            Console.ForegroundColor = ConsoleColor.Red;
                            Console.Write("o");
                            break;
                        case 2:
                            Console.ForegroundColor = ConsoleColor.Blue;
                            Console.Write("o");
                            break;
                    }
                    Console.ResetColor();
                }
                Console.WriteLine("║");
            }

            Console.Write(LowerRow);
            Console.WriteLine(LowerRow);
        }
    }
}
