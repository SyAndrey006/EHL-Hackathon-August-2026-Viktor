#!/usr/bin/env python3
"""Generate a dependency-free SVG comparing the three final policies."""
from pathlib import Path

POLICIES = [
    ("Logged control", 0.2524, 0.833, "No routing", "#150079", "circle"),
    ("Viktor heuristic", 0.0957, 0.820, "Length rule · −62.1%", "#999999", "square"),
    ("Smart router", 0.1126, 0.833, "Call-1 gate · −55.4%", "#6748FD", "diamond"),
]

def main():
    width, height = 1200, 680
    left, right, top, bottom = 120, 70, 85, 115
    xmin, xmax, ymin, ymax = .075, .270, .812, .840
    x = lambda v: left + (v-xmin)/(xmax-xmin)*(width-left-right)
    y = lambda v: height-bottom-(v-ymin)/(ymax-ymin)*(height-top-bottom)
    parts = [f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">
<title id="title">Three policies, one cache-aware trade-off</title>
<desc id="desc">Logged control costs 0.2524 dollars at quality 0.833. Viktor heuristic costs 0.0957 at quality 0.820. Smart router costs 0.1126 at quality 0.833.</desc>
<rect width="1200" height="680" fill="#fbfaff"/>
<style>text{{font-family:Inter,Segoe UI,sans-serif;fill:#150079}} .tick{{font-size:17px}} .label{{font-size:18px}} .name{{font-size:20px;font-weight:700}} .detail{{font-size:16px}}</style>
<text x="600" y="43" text-anchor="middle" font-size="30" font-weight="700">Three policies, one cache-aware trade-off</text>''']
    for value in [.10,.15,.20,.25]:
        px=x(value); parts.append(f'<line x1="{px:.1f}" y1="{top}" x2="{px:.1f}" y2="{height-bottom}" stroke="#150079" opacity=".10"/><text class="tick" x="{px:.1f}" y="{height-bottom+30}" text-anchor="middle">${value:.2f}</text>')
    for value in [.815,.820,.825,.830,.835,.840]:
        py=y(value); parts.append(f'<line x1="{left}" y1="{py:.1f}" x2="{width-right}" y2="{py:.1f}" stroke="#150079" opacity=".10"/><text class="tick" x="{left-18}" y="{py+6:.1f}" text-anchor="end">{value:.3f}</text>')
    parts += [f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#150079"/><line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#150079"/>',
              f'<text class="label" x="{(left+width-right)/2}" y="{height-55}" text-anchor="middle">Estimated input cost (USD, cache-aware)</text>',
              f'<text class="label" transform="translate(35 {(top+height-bottom)/2}) rotate(-90)" text-anchor="middle">Matched expected success</text>']
    # Viktor-to-smart trade-off connector.
    parts.append(f'<line x1="{x(.0957):.1f}" y1="{y(.820):.1f}" x2="{x(.1126):.1f}" y2="{y(.833):.1f}" stroke="#FFBD9E" stroke-width="7"/>')
    for name,cost,quality,detail,color,shape in POLICIES:
        px,py=x(cost),y(quality)
        if shape=="circle": mark=f'<circle cx="{px:.1f}" cy="{py:.1f}" r="12" fill="{color}"/>'
        elif shape=="square": mark=f'<rect x="{px-12:.1f}" y="{py-12:.1f}" width="24" height="24" fill="{color}"/>'
        else: mark=f'<path d="M {px:.1f} {py-15:.1f} L {px+15:.1f} {py:.1f} L {px:.1f} {py+15:.1f} L {px-15:.1f} {py:.1f} Z" fill="{color}"/>'
        parts.append(mark)
        if name=="Logged control": tx,ty,anchor=px-16,py+38,"end"
        elif name=="Viktor heuristic": tx,ty,anchor=px+20,py+40,"start"
        else: tx,ty,anchor=px+25,py-46,"start"
        parts.append(f'<text x="{tx:.1f}" y="{ty:.1f}" text-anchor="{anchor}"><tspan class="name" x="{tx:.1f}">{name}</tspan><tspan class="detail" x="{tx:.1f}" dy="24">${cost:.4f} · quality {quality:.3f}</tspan><tspan class="detail" x="{tx:.1f}" dy="21">{detail}</tspan></text>')
    parts += [f'<path d="M {x(.101):.1f} {y(.836):.1f} Q {x(.106):.1f} {y(.838):.1f} {x(.111):.1f} {y(.834):.1f}" fill="none" stroke="#6748FD" stroke-width="3" marker-end="url(#arrow)"/>',
              '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#6748FD"/></marker></defs>',
              f'<text x="{x(.101):.1f}" y="{y(.836)-10:.1f}" font-size="18" font-weight="700">+$0.0169 buys +0.013 quality</text>',
              '<text x="600" y="650" text-anchor="middle" font-size="16">Logged = control · Viktor = hindsight length heuristic · Smart = deployment-shaped Call-1 commitment</text></svg>']
    output=Path("results/policy_comparison.svg"); output.write_text("".join(parts),encoding="utf-8"); print(f"wrote {output}")

if __name__=="__main__": main()
