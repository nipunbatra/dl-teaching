-- Give the numerical planning ID a narrow column and the explanation more room.
function Table(tbl)
  local widths = {
    [3] = {0.06, 0.30, 0.64},
    [4] = {0.08, 0.26, 0.28, 0.38},
  }
  local selected = widths[#tbl.colspecs]
  if selected then
    for i, spec in ipairs(tbl.colspecs) do
      tbl.colspecs[i] = {
        i == 1 and pandoc.AlignRight or pandoc.AlignLeft,
        selected[i],
      }
    end
  end
  return tbl
end
